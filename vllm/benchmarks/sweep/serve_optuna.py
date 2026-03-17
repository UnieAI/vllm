# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import copy
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import requests

from vllm.utils.import_utils import PlaceholderModule

from .param_sweep import ParameterSweepItem
from .serve import SweepServeArgs, run_benchmark, run_server

try:
    import optuna
except ImportError:
    optuna = PlaceholderModule("optuna")


DEFAULT_VLLM_SEARCH_SPACE: dict[str, Any] = {
    "gpu_memory_utilization": {
        "type": "float",
        "low": 0.7,
        "high": 0.98,
        "step": 0.02,
    },
    "max_num_batched_tokens": {
        "type": "categorical",
        "choices": [None, 512, 1024, 2048, 4096, 8192],
    },
    "max_num_seqs": {
        "type": "categorical",
        "choices": [None, 8, 16, 32, 64, 128, 256],
    },
    "enable_chunked_prefill": {"type": "bool"},
    "enable_prefix_caching": {"type": "bool"},
}


def _require_optuna() -> None:
    if isinstance(optuna, PlaceholderModule):
        raise ImportError(
            "Please install optuna to use `vllm bench sweep serve_optuna`."
        )


def parse_score_concurrencies(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("score_concurrencies cannot be empty")
    if any(value <= 0 for value in values):
        raise ValueError("score_concurrencies must be > 0")
    return values


def read_search_space(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise TypeError("search space must be a JSON object")

    return loaded


def default_search_space() -> dict[str, Any]:
    # Return a deep copy so callers can safely mutate the object.
    return copy.deepcopy(DEFAULT_VLLM_SEARCH_SPACE)


def _extract_arg_value(cmd: list[str], flag: str) -> str | None:
    for i, token in enumerate(cmd):
        if token == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _has_flag(cmd: list[str], flag: str) -> bool:
    if flag in cmd:
        return True
    return any(token.startswith(flag + "=") for token in cmd)


def _extract_model_path_from_serve_cmd(serve_cmd: list[str]) -> str | None:
    model_arg = _extract_arg_value(serve_cmd, "--model")
    if model_arg:
        return model_arg

    for i, token in enumerate(serve_cmd):
        if token == "serve" and i + 1 < len(serve_cmd):
            candidate = serve_cmd[i + 1]
            if not candidate.startswith("-"):
                return candidate
    return None


def _read_model_max_len_from_config(model_path: str) -> int | None:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None

    try:
        with config_path.open("rb") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(loaded, dict):
        return None

    candidate_keys = (
        "max_model_len",
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "model_max_length",
    )

    for key in candidate_keys:
        value = loaded.get(key)
        if isinstance(value, int) and value > 0:
            return value

    text_config = loaded.get("text_config")
    if isinstance(text_config, dict):
        for key in candidate_keys:
            value = text_config.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return None


def _infer_max_model_len_from_serve_cmd(serve_cmd: list[str]) -> int | None:
    max_model_len_arg = _extract_arg_value(serve_cmd, "--max-model-len")
    if max_model_len_arg is not None:
        try:
            parsed = int(max_model_len_arg)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    model_path = _extract_model_path_from_serve_cmd(serve_cmd)
    if not model_path:
        return None

    return _read_model_max_len_from_config(model_path)


def _extract_served_model_name_from_serve_cmd(serve_cmd: list[str]) -> str | None:
    value = _extract_arg_value(serve_cmd, "--served-model-name")
    if value:
        return value
    return _extract_model_path_from_serve_cmd(serve_cmd)


def _extract_server_base_url_from_serve_cmd(serve_cmd: list[str]) -> str:
    host = _extract_arg_value(serve_cmd, "--host") or "127.0.0.1"
    port = _extract_arg_value(serve_cmd, "--port") or _extract_arg_value(serve_cmd, "-p")
    if port is None:
        port = "8000"

    # `vllm serve` often uses 0.0.0.0, but benchmark client should target localhost.
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _apply_default_bench_cmd_fields(
    bench_cmd: list[str],
    serve_cmd: list[str],
) -> list[str]:
    cmd = list(bench_cmd)
    if len(cmd) >= 3 and cmd[:3] == ["vllm", "bench", "serve"]:
        pass
    elif len(cmd) == 0:
        cmd = ["vllm", "bench", "serve"]

    model_name = _extract_served_model_name_from_serve_cmd(serve_cmd)
    model_path = _extract_model_path_from_serve_cmd(serve_cmd)
    base_url = _extract_server_base_url_from_serve_cmd(serve_cmd)

    if model_name and not _has_flag(cmd, "--model"):
        cmd.extend(["--model", model_name])
    if model_path and not _has_flag(cmd, "--tokenizer"):
        cmd.extend(["--tokenizer", model_path])
    if not _has_flag(cmd, "--base-url"):
        cmd.extend(["--base-url", base_url])
    if not _has_flag(cmd, "--backend"):
        cmd.extend(["--backend", "openai"])
    if not _has_flag(cmd, "--endpoint"):
        cmd.extend(["--endpoint", "/v1/completions"])
    return cmd


def _sanitize_serve_trial_params(
    trial_params: dict[str, Any],
    inferred_max_model_len: int | None,
) -> dict[str, Any]:
    sanitized = dict(trial_params)

    if inferred_max_model_len is None:
        return sanitized

    max_num_batched_tokens = sanitized.get("max_num_batched_tokens")
    if (
        isinstance(max_num_batched_tokens, int)
        and max_num_batched_tokens < inferred_max_model_len
    ):
        sanitized["max_num_batched_tokens"] = None

    return sanitized


def read_single_record(path: str | None) -> ParameterSweepItem:
    if path is None:
        return ParameterSweepItem()

    with open(path, "rb") as f:
        loaded = json.load(f)

    if isinstance(loaded, list):
        if len(loaded) != 1:
            raise ValueError("baseline_params JSON list must contain exactly one object")
        loaded = loaded[0]

    if not isinstance(loaded, dict):
        raise TypeError("baseline_params must be a JSON object")

    return ParameterSweepItem.from_record(loaded)


def suggest_trial_params(trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
    suggested: dict[str, Any] = {}

    for key, spec in search_space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            suggested[key] = spec
            continue

        dist_type = str(spec["type"])
        if dist_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"search space key '{key}' requires non-empty choices")
            suggested[key] = trial.suggest_categorical(key, choices)
            continue

        if dist_type == "bool":
            suggested[key] = trial.suggest_categorical(key, [True, False])
            continue

        if dist_type == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            log = bool(spec.get("log", False))
            suggested[key] = trial.suggest_int(key, low, high, step=step, log=log)
            continue

        if dist_type == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            step = spec.get("step")
            if step is not None:
                step = float(step)
            log = bool(spec.get("log", False))
            if step is not None and log:
                raise ValueError(
                    f"search space key '{key}' cannot set both step and log for float"
                )
            suggested[key] = trial.suggest_float(key, low, high, step=step, log=log)
            continue

        raise ValueError(f"unsupported distribution type for '{key}': {dist_type}")

    return suggested


def _default_params_from_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, spec in search_space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            defaults[key] = spec
            continue

        dist_type = str(spec["type"])
        if dist_type == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"search space key '{key}' requires non-empty choices")
            defaults[key] = choices[0]
        elif dist_type == "bool":
            defaults[key] = False
        elif dist_type == "int":
            defaults[key] = int(spec["low"])
        elif dist_type == "float":
            defaults[key] = float(spec["low"])
        else:
            raise ValueError(f"unsupported distribution type for '{key}': {dist_type}")

    return defaults


def _drop_none_values(params: dict[str, Any] | ParameterSweepItem) -> ParameterSweepItem:
    return ParameterSweepItem({k: v for k, v in dict(params).items() if v is not None})


def _start_best_server(
    serve_cmd: list[str],
    serve_overrides: ParameterSweepItem,
    *,
    show_stdout: bool,
    server_ready_timeout: int,
) -> int:
    best_server_cmd = serve_overrides.apply_to_cmd(serve_cmd)
    print("[START BEST SERVER]")
    print(f"Best server command: {best_server_cmd}")

    process = subprocess.Popen(
        best_server_cmd,
        start_new_session=True,
        stdout=None if show_stdout else subprocess.DEVNULL,
        stderr=None if show_stdout else subprocess.DEVNULL,
        env=os.environ | {"VLLM_SERVER_DEV_MODE": "1"},
    )

    start_time = time.monotonic()
    server_address = _extract_server_base_url_from_serve_cmd(best_server_cmd)
    health_url = server_address + "/health"
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"Best server process crashed with return code {process.returncode}"
            )
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                print(f"Best server is ready at {server_address} (pid={process.pid})")
                return process.pid
        except requests.RequestException:
            pass

        if time.monotonic() - start_time > server_ready_timeout:
            process.kill()
            raise TimeoutError(
                f"Best server failed to become ready within {server_ready_timeout} seconds"
            )
        time.sleep(1)


def score_benchmark_runs(
    run_data: list[dict[str, object]],
    score_metric: str,
    concurrency: int,
) -> tuple[float, float]:
    metric_values = list[float]()
    for run in run_data:
        if score_metric not in run:
            raise KeyError(f"benchmark output missing metric '{score_metric}'")
        metric_values.append(float(run[score_metric]))

    if not metric_values:
        raise RuntimeError("benchmark output is empty")

    metric_mean = sum(metric_values) / len(metric_values)
    return metric_mean / float(concurrency), metric_mean


def evaluate_configuration(
    serve_cmd: list[str],
    bench_cmd: list[str],
    after_bench_cmd: list[str],
    *,
    show_stdout: bool,
    dry_run: bool,
    server_ready_timeout: int,
    serve_overrides: ParameterSweepItem,
    bench_overrides: ParameterSweepItem,
    score_metric: str,
    score_concurrencies: list[int],
    num_runs: int,
    output_dir: Path,
) -> tuple[float, dict[str, Any]] | None:
    with run_server(
        serve_cmd,
        after_bench_cmd,
        show_stdout=show_stdout,
        serve_overrides=serve_overrides,
        dry_run=dry_run,
        server_ready_timeout=server_ready_timeout,
    ) as server:
        run_summaries = list[dict[str, Any]]()
        total_score = 0.0
        saw_dry_run_output = False

        for concurrency in score_concurrencies:
            benchmark_runs = list[dict[str, object]]()
            merged_bench_overrides = bench_overrides | {
                "max_concurrency": concurrency,
                "num_prompts": concurrency * 5,
            }

            for run_number in range(num_runs):
                run_data = run_benchmark(
                    server,
                    bench_cmd,
                    serve_overrides=serve_overrides,
                    bench_overrides=merged_bench_overrides,
                    run_number=run_number,
                    output_path=(
                        output_dir
                        / f"concurrency={concurrency}"
                        / f"run={run_number}.json"
                    ),
                    dry_run=dry_run,
                )
                if run_data is None:
                    assert dry_run
                    saw_dry_run_output = True
                    continue
                benchmark_runs.append(run_data)

            if saw_dry_run_output:
                continue

            normalized_score, metric_mean = score_benchmark_runs(
                benchmark_runs,
                score_metric,
                concurrency,
            )
            total_score += normalized_score
            run_summaries.append(
                {
                    "concurrency": concurrency,
                    "metric": score_metric,
                    "metric_mean": metric_mean,
                    "normalized_score": normalized_score,
                    "runs": benchmark_runs,
                }
            )

    if saw_dry_run_output:
        return None

    return total_score, {
        "score_formula": f"sum(mean({score_metric}) / concurrency)",
        "score_metric": score_metric,
        "score_concurrencies": score_concurrencies,
        "runs": run_summaries,
        "score": total_score,
    }


def append_trial_record(file_path: Path, record: dict[str, Any]) -> None:
    try:
        with file_path.open("rb") as f:
            loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        loaded = []

    if not isinstance(loaded, list):
        loaded = []

    loaded.append(record)

    with file_path.open("w") as f:
        json.dump(loaded, f, indent=4)


@dataclass
class SweepServeOptunaArgs(SweepServeArgs):
    search_space: dict[str, Any]
    n_trials: int
    direction: str
    score_metric: str
    score_concurrencies: list[int]
    baseline_params: ParameterSweepItem
    fixed_serve_overrides: ParameterSweepItem
    fixed_bench_overrides: ParameterSweepItem
    study_name: str | None
    sampler_seed: int | None
    start_best_server: bool

    parser_name: ClassVar[str] = "serve_optuna"
    parser_help: ClassVar[str] = "Tune serve parameters with Optuna."

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        base_args = SweepServeArgs.from_cli_args(args)
        base_args_dict = asdict(base_args)

        if len(base_args.serve_params) != 1:
            raise ValueError(
                "serve_optuna supports exactly one fixed serve_params entry. "
                "Use --search-space to tune values."
            )
        if len(base_args.bench_params) != 1:
            raise ValueError(
                "serve_optuna supports exactly one fixed bench_params entry. "
                "Use --score-concurrencies for benchmark concurrency sweep."
            )
        if base_args.link_vars:
            raise ValueError("serve_optuna does not support --link-vars")

        if args.n_trials < 1:
            raise ValueError("n_trials should be at least 1")

        base_args_dict["bench_cmd"] = _apply_default_bench_cmd_fields(
            base_args.bench_cmd,
            base_args.serve_cmd,
        )

        return cls(
            **base_args_dict,
            search_space=(
                default_search_space()
                if args.search_space is None
                else read_search_space(args.search_space)
            ),
            n_trials=args.n_trials,
            direction=args.direction,
            score_metric=args.score_metric,
            score_concurrencies=parse_score_concurrencies(args.score_concurrencies),
            baseline_params=read_single_record(args.baseline_params),
            fixed_serve_overrides=base_args.serve_params[0],
            fixed_bench_overrides=base_args.bench_params[0],
            study_name=args.study_name,
            sampler_seed=args.sampler_seed,
            start_best_server=args.start_best_server,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().add_cli_args(parser)
        parser.set_defaults(num_runs=1)

        optuna_group = parser.add_argument_group("optuna options")
        optuna_group.add_argument(
            "--search-space",
            type=str,
            default=None,
            help=(
                "Optional path to JSON object defining trial parameters. "
                "Each key supports one of: "
                "{\"type\":\"float\",\"low\":...,\"high\":...,\"step\":...}, "
                "{\"type\":\"int\",\"low\":...,\"high\":...,\"step\":...}, "
                "{\"type\":\"categorical\",\"choices\":[...]}, "
                "{\"type\":\"bool\"}. "
                "Non-object values are treated as constants. "
                "If omitted, uses built-in vLLM defaults for common "
                "serve tuning knobs."
            ),
        )
        optuna_group.add_argument(
            "--n-trials",
            type=int,
            default=20,
            help="Number of Optuna trials.",
        )
        optuna_group.add_argument(
            "--direction",
            type=str,
            choices=("maximize", "minimize"),
            default="maximize",
            help="Optimization direction for score.",
        )
        optuna_group.add_argument(
            "--score-metric",
            type=str,
            default="total_token_throughput",
            help="Metric key read from each benchmark result JSON.",
        )
        optuna_group.add_argument(
            "--score-concurrencies",
            type=str,
            default="1,8,64,256",
            help="Comma-separated concurrency list used for scoring.",
        )
        optuna_group.add_argument(
            "--baseline-params",
            type=str,
            default=None,
            help=(
                "Optional path to JSON object of serve overrides for baseline run. "
                "If omitted, baseline uses the base serve command plus fixed serve_params."
            ),
        )
        optuna_group.add_argument(
            "--study-name",
            type=str,
            default=None,
            help="Optional Optuna study name.",
        )
        optuna_group.add_argument(
            "--sampler-seed",
            type=int,
            default=None,
            help="Optional random seed for Optuna sampler.",
        )
        optuna_group.add_argument(
            "--start-best-server",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Whether to launch `serve_cmd` with the best Optuna parameters "
                "after optimization."
            ),
        )

        return parser


def run_main(args: SweepServeOptunaArgs):
    _require_optuna()
    inferred_max_model_len = _infer_max_model_len_from_serve_cmd(args.serve_cmd)

    timestamp = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp

    if args.resume and not output_dir.exists():
        raise ValueError(f"Cannot resume from non-existent directory ({output_dir})")

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = output_dir / "baseline.json"
    trials_file = output_dir / "trials.json"
    best_params_file = output_dir / "best_params.json"
    best_file = output_dir / "best.json"

    baseline_overrides = _drop_none_values(args.fixed_serve_overrides | args.baseline_params)

    baseline_result = evaluate_configuration(
        args.serve_cmd,
        args.bench_cmd,
        args.after_bench_cmd,
        show_stdout=args.show_stdout,
        dry_run=args.dry_run,
        server_ready_timeout=args.server_ready_timeout,
        serve_overrides=baseline_overrides,
        bench_overrides=args.fixed_bench_overrides,
        score_metric=args.score_metric,
        score_concurrencies=args.score_concurrencies,
        num_runs=args.num_runs,
        output_dir=output_dir / "baseline_runs",
    )

    if baseline_result is None:
        assert args.dry_run
        preview_overrides = _drop_none_values(
            args.fixed_serve_overrides | _default_params_from_search_space(args.search_space)
        )
        evaluate_configuration(
            args.serve_cmd,
            args.bench_cmd,
            args.after_bench_cmd,
            show_stdout=args.show_stdout,
            dry_run=True,
            server_ready_timeout=args.server_ready_timeout,
            serve_overrides=preview_overrides,
            bench_overrides=args.fixed_bench_overrides,
            score_metric=args.score_metric,
            score_concurrencies=args.score_concurrencies,
            num_runs=args.num_runs,
            output_dir=output_dir / "sample_trial",
        )
        return None

    baseline_score, baseline_payload = baseline_result
    baseline_record = {
        "params": dict(baseline_overrides),
        "score": baseline_score,
        "result": baseline_payload,
    }
    with baseline_file.open("w") as f:
        json.dump(baseline_record, f, indent=4)

    append_trial_record(
        trials_file,
        {
            "trial": -1,
            "state": "baseline",
            "params": dict(baseline_overrides),
            "score": baseline_score,
            "result": baseline_payload,
        },
    )

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
    )

    def objective(trial: Any) -> float:
        trial_params = _sanitize_serve_trial_params(
            suggest_trial_params(trial, args.search_space),
            inferred_max_model_len,
        )
        trial.set_user_attr("effective_serve_params", trial_params)
        serve_overrides = _drop_none_values(args.fixed_serve_overrides | trial_params)

        try:
            evaluated = evaluate_configuration(
                args.serve_cmd,
                args.bench_cmd,
                args.after_bench_cmd,
                show_stdout=args.show_stdout,
                dry_run=False,
                server_ready_timeout=args.server_ready_timeout,
                serve_overrides=serve_overrides,
                bench_overrides=args.fixed_bench_overrides,
                score_metric=args.score_metric,
                score_concurrencies=args.score_concurrencies,
                num_runs=args.num_runs,
                output_dir=output_dir / f"trial={trial.number}",
            )
            if evaluated is None:
                raise RuntimeError("unexpected dry-run state during optimization")

            score, payload = evaluated
            trial.set_user_attr("benchmark_result", payload)
            append_trial_record(
                trials_file,
                {
                    "trial": trial.number,
                    "state": "complete",
                    "params": trial_params,
                    "score": score,
                    "result": payload,
                },
            )
            return score
        except BaseException as exc:
            append_trial_record(
                trials_file,
                {
                    "trial": trial.number,
                    "state": "pruned",
                    "params": trial_params,
                    "score": None,
                    "error": str(exc),
                },
            )
            raise optuna.TrialPruned() from exc

    study.optimize(objective, n_trials=args.n_trials)

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("No completed trials. All trials were pruned.")

    best_trial = study.best_trial
    best_payload = best_trial.user_attrs.get("benchmark_result")
    if not isinstance(best_payload, dict):
        raise RuntimeError("best trial is missing benchmark result payload")

    effective_best_params = best_trial.user_attrs.get("effective_serve_params")
    if not isinstance(effective_best_params, dict):
        effective_best_params = _sanitize_serve_trial_params(
            dict(best_trial.params),
            inferred_max_model_len,
        )

    with best_params_file.open("w") as f:
        json.dump(effective_best_params, f, indent=4)

    best_record = {
        "trial": best_trial.number,
        "params": effective_best_params,
        "score": best_trial.value,
        "result": best_payload,
    }
    with best_file.open("w") as f:
        json.dump(best_record, f, indent=4)

    if args.start_best_server:
        effective_best_overrides = _drop_none_values(
            args.fixed_serve_overrides | effective_best_params
        )
        _start_best_server(
            args.serve_cmd,
            effective_best_overrides,
            show_stdout=args.show_stdout,
            server_ready_timeout=args.server_ready_timeout,
        )

    return best_record


def main(args: argparse.Namespace):
    run_main(SweepServeOptunaArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepServeOptunaArgs.parser_help)
    SweepServeOptunaArgs.add_cli_args(parser)

    main(parser.parse_args())
