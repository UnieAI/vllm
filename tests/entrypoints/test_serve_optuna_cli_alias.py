# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.cli.serve_optuna import ServeOptunaSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _build_parser() -> tuple[FlexibleArgumentParser, ServeOptunaSubcommand]:
    parser = FlexibleArgumentParser(description="vLLM CLI test parser")
    subparsers = parser.add_subparsers(required=True, dest="subparser")
    cmd = ServeOptunaSubcommand()
    cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
    return parser, cmd


def test_serve_optuna_alias_dispatches_to_sweep_entrypoint(monkeypatch, tmp_path):
    calls = {"count": 0}

    def _fake_main(args):
        calls["count"] += 1
        calls["subparser"] = args.subparser

    monkeypatch.setattr(
        "vllm.entrypoints.cli.serve_optuna.serve_optuna_main",
        _fake_main,
    )

    search_space_file = tmp_path / "search_space.json"
    search_space_file.write_text("{}", encoding="utf-8")

    parser, _ = _build_parser()
    args = parser.parse_args(
        [
            "serve-optuna",
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B",
            "--bench-cmd",
            "vllm bench serve --model Qwen/Qwen3-0.6B",
            "--search-space",
            str(search_space_file),
            "--dry-run",
        ]
    )
    args.dispatch_function(args)

    assert calls["count"] == 1
    assert calls["subparser"] == "serve-optuna"


def test_serve_optuna_underscore_alias_is_supported(tmp_path):
    search_space_file = tmp_path / "search_space.json"
    search_space_file.write_text("{}", encoding="utf-8")

    parser, _ = _build_parser()
    args = parser.parse_args(
        [
            "serve_optuna",
            "--serve-cmd",
            "vllm serve Qwen/Qwen3-0.6B",
            "--bench-cmd",
            "vllm bench serve --model Qwen/Qwen3-0.6B",
            "--search-space",
            str(search_space_file),
            "--dry-run",
        ]
    )

    assert args.subparser in {"serve-optuna", "serve_optuna"}
