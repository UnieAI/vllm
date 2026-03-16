# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from types import SimpleNamespace

import numpy as np
from benchmark_utils import TimeCollector
from tabulate import tabulate

from vllm.config import SpeculativeConfig
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.spec_decode.ngram_proposer import NgramProposer, NgramProposalInputs


def _build_proposer(
    args,
    *,
    max_ngram: int,
    max_model_len: int | None = None,
) -> NgramProposer:
    return NgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=max_model_len
                or (args.num_token + args.num_spec_token),
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=args.num_req),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=args.min_ngram,
                prompt_lookup_max=max_ngram,
                prompt_lookup_window=args.search_window,
                num_speculative_tokens=args.num_spec_token,
                method="ngram",
            ),
        )
    )


def benchmark_propose(args):
    rows = []
    for max_ngram in args.max_ngram:
        filter_collector = TimeCollector(TimeCollector.US)
        match_collector = TimeCollector(TimeCollector.US)
        materialize_collector = TimeCollector(TimeCollector.US)
        propose_collector = TimeCollector(TimeCollector.US)
        proposer = _build_proposer(args, max_ngram=max_ngram)
        sampled_token_ids = [[0] for _ in range(args.num_req)]
        num_tokens_no_spec = np.full(args.num_req, args.num_token, dtype=np.int32)

        gc.collect()
        for _ in range(args.num_iteration):
            token_ids_cpu = np.random.randint(
                0,
                20,
                (args.num_req, args.num_token),
                dtype=np.int32,
            )
            with filter_collector:
                valid_ngram_requests = proposer.get_valid_ngram_requests(
                    sampled_token_ids,
                    num_tokens_no_spec,
                )
            with match_collector:
                proposer.run_batch_match(
                    valid_ngram_requests,
                    num_tokens_no_spec,
                    token_ids_cpu,
                )
            with materialize_collector:
                proposer.materialize_draft_token_ids(
                    args.num_req,
                    valid_ngram_requests,
                )
            with propose_collector:
                proposer.propose(
                    sampled_token_ids,
                    num_tokens_no_spec,
                    token_ids_cpu,
                )
        rows.append(
            [
                args.num_req,
                args.num_token,
                args.min_ngram,
                max_ngram,
                args.search_window or "full",
                *filter_collector.dump_avg_max(),
                *match_collector.dump_avg_max(),
                *materialize_collector.dump_avg_max(),
                *propose_collector.dump_avg_max(),
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "# Request",
                "# Token",
                "Min Ngram",
                "Max Ngram",
                "Search Window",
                "Filter Avg (us)",
                "Filter Max (us)",
                "Match Avg (us)",
                "Match Max (us)",
                "Materialize Avg (us)",
                "Materialize Max (us)",
                "Propose Avg (us)",
                "Propose Max (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def benchmark_batched_propose(args):
    class FakeNgramRunner:
        def __init__(self, proposer: NgramProposer, num_req: int, num_token: int):
            self.drafter = proposer
            self.max_model_len = proposer.max_model_len
            self.input_batch = SimpleNamespace(
                num_tokens_no_spec=np.full(num_req, num_token, dtype=np.int32),
                token_ids_cpu=np.random.randint(
                    0,
                    20,
                    (num_req, num_token),
                    dtype=np.int32,
                ),
            )

        def get_ngram_proposal_inputs(
            self,
            sampled_token_ids: list[list[int]],
        ) -> NgramProposalInputs:
            valid_ngram_requests = np.empty(len(sampled_token_ids), dtype=np.int32)
            num_valid_requests = 0
            for i, sampled_ids in enumerate(sampled_token_ids):
                if self.drafter.should_propose_for_request(
                    i,
                    sampled_ids,
                    self.input_batch.num_tokens_no_spec[i],
                ):
                    valid_ngram_requests[num_valid_requests] = i
                    num_valid_requests += 1

            return NgramProposalInputs(
                sampled_token_ids=sampled_token_ids,
                num_tokens_no_spec=self.input_batch.num_tokens_no_spec,
                token_ids_cpu=self.input_batch.token_ids_cpu,
                valid_ngram_requests=valid_ngram_requests[:num_valid_requests],
            )

        def propose_ngram_draft_token_ids(
            self,
            sampled_token_ids: list[list[int]],
        ) -> list[list[int]]:
            ngram_inputs = self.get_ngram_proposal_inputs(sampled_token_ids)
            return self.drafter.propose(
                ngram_inputs.sampled_token_ids,
                ngram_inputs.num_tokens_no_spec,
                ngram_inputs.token_ids_cpu,
                valid_ngram_requests=ngram_inputs.valid_ngram_requests,
            )

    proposer = _build_proposer(
        args,
        max_ngram=max(args.max_ngram),
        max_model_len=args.num_token + args.num_spec_token,
    )
    runner = FakeNgramRunner(proposer, args.num_req, args.num_token)
    sampled_token_ids = [[0]] * args.num_req
    inputs_collector = TimeCollector(TimeCollector.US)
    propose_collector = TimeCollector(TimeCollector.US)

    print("Starting benchmark")
    for _ in range(args.num_iteration):
        with inputs_collector:
            runner.get_ngram_proposal_inputs(sampled_token_ids)
        with propose_collector:
            runner.propose_ngram_draft_token_ids(sampled_token_ids)

    rows = [[
        args.num_req,
        args.num_token,
        args.search_window or "full",
        inputs_collector.avg(),
        inputs_collector.max(),
        propose_collector.avg(),
        propose_collector.max(),
    ]]
    print(
        tabulate(
            rows,
            headers=[
                "# Request",
                "# Token",
                "Search Window",
                "Input Avg (us)",
                "Input Max (us)",
                "Runner Propose Avg (us)",
                "Runner Propose Max (us)",
            ],
            tablefmt="grid",
            floatfmt=".3f",
        )
    )


def invoke_main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the performance of N-gram speculative decode drafting"
    )
    parser.add_argument(
        "--batched", action="store_true", help="consider time to prepare batch"
    )
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=100,
        help="Number of iterations to run to stabilize final data readings",
    )
    parser.add_argument(
        "--num-req", type=int, default=128, help="Number of requests in the batch"
    )
    parser.add_argument(
        "--num-token", type=int, default=1500, help="Number of tokens for each request"
    )
    parser.add_argument(
        "--min-ngram",
        type=int,
        default=3,
        help="Minimum n-gram to match",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        nargs="*",
        default=[5, 7, 10, 15, 20],
        help="Maximum n-gram to match",
    )
    parser.add_argument(
        "--search-window",
        type=int,
        default=None,
        help="Only search the most recent N context tokens. Unset means full "
        "history.",
    )
    parser.add_argument(
        "--num-spec-token",
        type=int,
        default=3,
        help="Number of speculative tokens to generate",
    )
    args = parser.parse_args()

    if not args.batched:
        benchmark_propose(args)
    else:
        benchmark_batched_propose(args)


"""
# Example command lines:
# time python3 benchmarks/benchmark_ngram_proposer.py
# time python3 benchmarks/benchmark_ngram_proposer.py --batched --num-iteration 4 --num-token 1000000 --num-req 128
"""  # noqa: E501
if __name__ == "__main__":
    invoke_main()  # pragma: no cover
