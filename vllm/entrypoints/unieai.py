# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import asyncio
try:
    import uvloop
except ImportError:
    uvloop = None
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.logger import init_logger

logger = init_logger(__name__)

def main():
    parser = FlexibleArgumentParser(description="UnieAI CLI - Wrapper for vLLM activities")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: openai-api-server
    # This forwards all arguments to vllm.entrypoints.openai.api_server
    openai_parser = subparsers.add_parser("openai-api-server", 
                                        help="Launch the OpenAI-compatible API server")
    
    # We reuse the vLLM argument parser to ensure all flags are available
    make_arg_parser(openai_parser)

    # UnieAI specific features can be added here as new subcommands or flags
    # example_parser = subparsers.add_parser("feature-x", help="Custom UnieAI feature")

    args = parser.parse_args()

    if args.subcommand == "openai-api-server":
        logger.info("Routing launch activity to vllm.entrypoints.openai.api_server")
        validate_parsed_serve_args(args)
        
        # vLLM's run_server is an async function
        if uvloop is not None:
            uvloop.run(run_server(args))
        else:
            asyncio.run(run_server(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
