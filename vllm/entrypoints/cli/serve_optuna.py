# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

from vllm.benchmarks.sweep.serve_optuna import SweepServeOptunaArgs
from vllm.benchmarks.sweep.serve_optuna import main as serve_optuna_main
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG

if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class ServeOptunaSubcommand(CLISubcommand):
    """The `serve-optuna` top-level subcommand for the vLLM CLI."""

    name = "serve-optuna"
    help = "Tune vLLM serve parameters with Optuna."

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        serve_optuna_main(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            aliases=["serve_optuna"],
            help=self.help,
            description=self.help,
            usage=f"vllm {self.name} [options]",
        )
        SweepServeOptunaArgs.add_cli_args(parser)
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeOptunaSubcommand()]
