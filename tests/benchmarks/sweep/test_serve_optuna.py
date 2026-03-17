# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import argparse
import contextlib
from types import SimpleNamespace

import pytest

from vllm.benchmarks.sweep.param_sweep import ParameterSweep
from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep import serve_optuna
from vllm.benchmarks.sweep.serve_optuna import (
    SweepServeOptunaArgs,
    default_search_space,
    parse_score_concurrencies,
    run_main,
    suggest_trial_params,
)


def test_parse_score_concurrencies():
    assert parse_score_concurrencies("1,8,64,256") == [1, 8, 64, 256]


def test_parse_score_concurrencies_invalid():
    with pytest.raises(ValueError):
        parse_score_concurrencies("1,0")


def test_suggest_trial_params():
    search_space = {
        "gpu_memory_utilization": {
            "type": "float",
            "low": 0.7,
            "high": 0.98,
            "step": 0.02,
        },
        "max_num_seqs": {"type": "int", "low": 8, "high": 32, "step": 8},
        "max_num_batched_tokens": {
            "type": "categorical",
            "choices": [None, 1024, 2048],
        },
        "enable_prefix_caching": {"type": "bool"},
        "constant_param": "keep",
    }

    class DummyTrial:
        def suggest_float(self, name, low, high, step=None, log=False):  # noqa: ARG002
            assert name == "gpu_memory_utilization"
            return 0.78

        def suggest_int(self, name, low, high, step=1, log=False):  # noqa: ARG002
            assert name == "max_num_seqs"
            return 24

        def suggest_categorical(self, name, choices):  # noqa: ARG002
            if name == "max_num_batched_tokens":
                return 1024
            assert name == "enable_prefix_caching"
            return True

    trial = DummyTrial()

    suggested = suggest_trial_params(trial, search_space)

    assert suggested == {
        "gpu_memory_utilization": 0.78,
        "max_num_seqs": 24,
        "max_num_batched_tokens": 1024,
        "enable_prefix_caching": True,
        "constant_param": "keep",
    }


def test_from_cli_args_uses_default_search_space_when_omitted():
    parser = argparse.ArgumentParser()
    SweepServeOptunaArgs.add_cli_args(parser)

    parsed = parser.parse_args(
        [
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B",
            "--bench-cmd",
            "vllm bench serve --model Qwen/Qwen3-0.6B",
            "--dry-run",
        ]
    )

    args = SweepServeOptunaArgs.from_cli_args(parsed)
    assert args.search_space == default_search_space()
    assert args.start_best_server is True


def test_from_cli_args_reads_search_space_file(tmp_path):
    parser = argparse.ArgumentParser()
    SweepServeOptunaArgs.add_cli_args(parser)

    search_space_path = tmp_path / "search_space.json"
    payload = {"max_num_seqs": {"type": "categorical", "choices": [16, 32, 64]}}
    search_space_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = parser.parse_args(
        [
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B",
            "--bench-cmd",
            "vllm bench serve --model Qwen/Qwen3-0.6B",
            "--search-space",
            str(search_space_path),
            "--dry-run",
        ]
    )

    args = SweepServeOptunaArgs.from_cli_args(parsed)
    assert args.search_space == payload


def test_from_cli_args_populates_bench_cmd_from_serve_cmd():
    parser = argparse.ArgumentParser()
    SweepServeOptunaArgs.add_cli_args(parser)

    parsed = parser.parse_args(
        [
            "--serve-cmd",
            "vllm serve /tmp/model --served-model-name test --host 0.0.0.0 --port 12470",
            "--dry-run",
        ]
    )

    args = SweepServeOptunaArgs.from_cli_args(parsed)
    assert args.bench_cmd[0:3] == ["vllm", "bench", "serve"]
    assert "--model" in args.bench_cmd
    assert "--base-url" in args.bench_cmd
    assert "http://127.0.0.1:12470" in args.bench_cmd


def test_drop_none_values_for_serve_overrides():
    cleaned = serve_optuna._drop_none_values(
        ParameterSweepItem(
            {
                "gpu_memory_utilization": 0.9,
                "max_num_batched_tokens": None,
                "max_num_seqs": 64,
            }
        )
    )
    assert cleaned == {
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 64,
    }


def test_infer_max_model_len_from_serve_cmd_reads_flag():
    serve_cmd = [
        "vllm",
        "serve",
        "Qwen/Qwen3-0.6B",
        "--max-model-len",
        "8192",
    ]
    assert serve_optuna._infer_max_model_len_from_serve_cmd(serve_cmd) == 8192


def test_sanitize_serve_trial_params_drops_invalid_max_num_batched_tokens():
    params = {
        "gpu_memory_utilization": 0.98,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 32,
    }
    sanitized = serve_optuna._sanitize_serve_trial_params(
        params,
        inferred_max_model_len=40960,
    )
    assert sanitized["max_num_batched_tokens"] is None
    assert sanitized["gpu_memory_utilization"] == 0.98
    assert sanitized["max_num_seqs"] == 32


def test_run_main_writes_outputs(tmp_path, monkeypatch):
    class FakeTrialState:
        COMPLETE = "complete"

    class FakeTrial:
        def __init__(self, number: int):
            self.number = number
            self.params: dict[str, object] = {}
            self.user_attrs: dict[str, object] = {}
            self.state = FakeTrialState.COMPLETE
            self.value: float | None = None

        def suggest_float(self, name, low, high, step=None, log=False):  # noqa: ARG002
            value = low
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high, step=1, log=False):  # noqa: ARG002
            value = low
            self.params[name] = value
            return value

        def suggest_categorical(self, name, choices):
            value = choices[0]
            self.params[name] = value
            return value

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class FakeStudy:
        def __init__(self, direction: str):
            self.direction = direction
            self.trials: list[FakeTrial] = []
            self.best_trial: FakeTrial | None = None

        def optimize(self, objective, n_trials: int):
            for trial_number in range(n_trials):
                trial = FakeTrial(trial_number)
                trial.value = objective(trial)
                self.trials.append(trial)

            if self.direction == "minimize":
                self.best_trial = min(self.trials, key=lambda trial: trial.value or 0.0)
            else:
                self.best_trial = max(self.trials, key=lambda trial: trial.value or 0.0)

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda seed=None: object()),  # noqa: ARG005
        trial=SimpleNamespace(TrialState=FakeTrialState),
        TrialPruned=RuntimeError,
        create_study=lambda **kwargs: FakeStudy(kwargs["direction"]),
    )
    monkeypatch.setattr(serve_optuna, "optuna", fake_optuna)

    def mock_evaluate_configuration(*args, **kwargs):
        output_dir = kwargs["output_dir"]
        if output_dir.name == "baseline_runs":
            return 10.0, {"score": 10.0, "runs": []}
        if output_dir.name.startswith("trial="):
            return 12.0, {"score": 12.0, "runs": []}
        raise AssertionError(f"unexpected output dir: {output_dir}")

    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_optuna.evaluate_configuration",
        mock_evaluate_configuration,
    )

    args = SweepServeOptunaArgs(
        serve_cmd=["vllm", "serve", "Qwen/Qwen3-0.6B"],
        bench_cmd=["vllm", "bench", "serve", "--model", "Qwen/Qwen3-0.6B"],
        after_bench_cmd=[],
        show_stdout=False,
        serve_params=ParameterSweep.from_records([{}]),
        bench_params=ParameterSweep.from_records([{}]),
        output_dir=tmp_path,
        num_runs=1,
        dry_run=False,
        resume=None,
        link_vars=[],
        server_ready_timeout=1,
        search_space={},
        n_trials=1,
        direction="maximize",
        score_metric="total_token_throughput",
        score_concurrencies=[1, 8],
        baseline_params=ParameterSweepItem(),
        fixed_serve_overrides=ParameterSweepItem(),
        fixed_bench_overrides=ParameterSweepItem(),
        study_name="test-study",
        sampler_seed=0,
        start_best_server=False,
    )

    best_record = run_main(args)

    assert best_record is not None
    assert best_record["score"] == 12.0

    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    with (run_dir / "baseline.json").open("rb") as f:
        baseline = json.load(f)
    with (run_dir / "best.json").open("rb") as f:
        best = json.load(f)
    with (run_dir / "best_params.json").open("rb") as f:
        best_params = json.load(f)
    with (run_dir / "trials.json").open("rb") as f:
        trials = json.load(f)

    assert baseline["score"] == 10.0
    assert best["score"] == 12.0
    assert best_params == {}
    assert len(trials) == 2
    assert trials[0]["state"] == "baseline"
    assert trials[1]["state"] == "complete"


def test_run_main_starts_best_server_when_enabled(tmp_path, monkeypatch):
    class FakeTrialState:
        COMPLETE = "complete"

    class FakeTrial:
        def __init__(self, number: int):
            self.number = number
            self.params = {"gpu_memory_utilization": 0.9}
            self.user_attrs: dict[str, object] = {}
            self.state = FakeTrialState.COMPLETE
            self.value = 1.0

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class FakeStudy:
        def __init__(self, direction: str):  # noqa: ARG002
            self.trials = [FakeTrial(0)]
            self.best_trial = self.trials[0]

        def optimize(self, objective, n_trials: int):  # noqa: ARG002
            trial = self.trials[0]
            trial.value = objective(trial)

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda seed=None: object()),  # noqa: ARG005
        trial=SimpleNamespace(TrialState=FakeTrialState),
        TrialPruned=RuntimeError,
        create_study=lambda **kwargs: FakeStudy(kwargs["direction"]),
    )
    monkeypatch.setattr(serve_optuna, "optuna", fake_optuna)

    def mock_evaluate_configuration(*args, **kwargs):  # noqa: ARG001
        return 1.0, {"score": 1.0, "runs": []}

    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_optuna.evaluate_configuration",
        mock_evaluate_configuration,
    )

    start_calls = {"count": 0}

    def mock_start_best_server(*args, **kwargs):  # noqa: ARG001
        start_calls["count"] += 1
        return 12345

    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_optuna._start_best_server",
        mock_start_best_server,
    )

    args = SweepServeOptunaArgs(
        serve_cmd=["vllm", "serve", "Qwen/Qwen3-0.6B"],
        bench_cmd=["vllm", "bench", "serve", "--model", "Qwen/Qwen3-0.6B"],
        after_bench_cmd=[],
        show_stdout=False,
        serve_params=ParameterSweep.from_records([{}]),
        bench_params=ParameterSweep.from_records([{}]),
        output_dir=tmp_path,
        num_runs=1,
        dry_run=False,
        resume=None,
        link_vars=[],
        server_ready_timeout=1,
        search_space={},
        n_trials=1,
        direction="maximize",
        score_metric="total_token_throughput",
        score_concurrencies=[1],
        baseline_params=ParameterSweepItem(),
        fixed_serve_overrides=ParameterSweepItem(),
        fixed_bench_overrides=ParameterSweepItem(),
        study_name="test-study",
        sampler_seed=0,
        start_best_server=True,
    )

    best_record = run_main(args)
    assert best_record is not None
    assert start_calls["count"] == 1


def test_evaluate_configuration_sets_num_prompts_from_concurrency(
    tmp_path,
    monkeypatch,
):
    captured_bench_overrides: list[dict[str, object]] = []

    @contextlib.contextmanager
    def mock_run_server(*args, **kwargs):  # noqa: ARG001
        yield object()

    def mock_run_benchmark(
        server,  # noqa: ARG001
        bench_cmd,  # noqa: ARG001
        *,
        serve_overrides,  # noqa: ARG001
        bench_overrides,
        run_number,  # noqa: ARG001
        output_path,  # noqa: ARG001
        dry_run,  # noqa: ARG001
    ):
        captured_bench_overrides.append(dict(bench_overrides))
        return {"total_token_throughput": 100.0}

    monkeypatch.setattr("vllm.benchmarks.sweep.serve_optuna.run_server", mock_run_server)
    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_optuna.run_benchmark",
        mock_run_benchmark,
    )

    result = serve_optuna.evaluate_configuration(
        serve_cmd=["vllm", "serve", "Qwen/Qwen3-0.6B"],
        bench_cmd=["vllm", "bench", "serve", "--model", "Qwen/Qwen3-0.6B"],
        after_bench_cmd=[],
        show_stdout=False,
        dry_run=False,
        server_ready_timeout=1,
        serve_overrides=ParameterSweepItem(),
        bench_overrides=ParameterSweepItem({"request_rate": 1.0}),
        score_metric="total_token_throughput",
        score_concurrencies=[1, 4],
        num_runs=1,
        output_dir=tmp_path,
    )

    assert result is not None
    assert len(captured_bench_overrides) == 2
    assert captured_bench_overrides[0]["max_concurrency"] == 1
    assert captured_bench_overrides[0]["num_prompts"] == 5
    assert captured_bench_overrides[1]["max_concurrency"] == 4
    assert captured_bench_overrides[1]["num_prompts"] == 20


def test_run_main_starts_best_server_with_effective_sanitized_params(
    tmp_path,
    monkeypatch,
):
    class FakeTrialState:
        COMPLETE = "complete"

    class FakeTrial:
        def __init__(self, number: int):
            self.number = number
            self.params: dict[str, object] = {}
            self.user_attrs: dict[str, object] = {}
            self.state = FakeTrialState.COMPLETE
            self.value: float | None = None

        def suggest_categorical(self, name, choices):  # noqa: ARG002
            value = 1024
            self.params[name] = value
            return value

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    class FakeStudy:
        def __init__(self, direction: str):  # noqa: ARG002
            self.trials: list[FakeTrial] = []
            self.best_trial: FakeTrial | None = None

        def optimize(self, objective, n_trials: int):  # noqa: ARG002
            trial = FakeTrial(0)
            trial.value = objective(trial)
            self.trials = [trial]
            self.best_trial = trial

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda seed=None: object()),  # noqa: ARG005
        trial=SimpleNamespace(TrialState=FakeTrialState),
        TrialPruned=RuntimeError,
        create_study=lambda **kwargs: FakeStudy(kwargs["direction"]),
    )
    monkeypatch.setattr(serve_optuna, "optuna", fake_optuna)

    def mock_evaluate_configuration(*args, **kwargs):  # noqa: ARG001
        return 1.0, {"score": 1.0, "runs": []}

    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_optuna.evaluate_configuration",
        mock_evaluate_configuration,
    )

    start_call: dict[str, object] = {}

    def mock_start_best_server(serve_cmd, serve_overrides, **kwargs):  # noqa: ARG001
        start_call["serve_cmd"] = serve_cmd
        start_call["serve_overrides"] = dict(serve_overrides)
        return 12345

    monkeypatch.setattr(
        "vllm.benchmarks.sweep.serve_optuna._start_best_server",
        mock_start_best_server,
    )

    args = SweepServeOptunaArgs(
        serve_cmd=[
            "vllm",
            "serve",
            "Qwen/Qwen3-0.6B",
            "--max-model-len",
            "40960",
        ],
        bench_cmd=["vllm", "bench", "serve", "--model", "Qwen/Qwen3-0.6B"],
        after_bench_cmd=[],
        show_stdout=False,
        serve_params=ParameterSweep.from_records([{}]),
        bench_params=ParameterSweep.from_records([{}]),
        output_dir=tmp_path,
        num_runs=1,
        dry_run=False,
        resume=None,
        link_vars=[],
        server_ready_timeout=1,
        search_space={
            "max_num_batched_tokens": {"type": "categorical", "choices": [1024]}
        },
        n_trials=1,
        direction="maximize",
        score_metric="total_token_throughput",
        score_concurrencies=[1],
        baseline_params=ParameterSweepItem(),
        fixed_serve_overrides=ParameterSweepItem(),
        fixed_bench_overrides=ParameterSweepItem(),
        study_name="test-study",
        sampler_seed=0,
        start_best_server=True,
    )

    best_record = run_main(args)

    assert best_record is not None
    assert best_record["params"]["max_num_batched_tokens"] is None
    assert "serve_overrides" in start_call
    # None-valued overrides must not be emitted into the final serve command.
    assert "max_num_batched_tokens" not in start_call["serve_overrides"]
