# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Utilities to start disaggregated serving on qaic devices."""
import argparse
import re
import os
import copy
import dataclasses
import ssl
import sys
import time
import subprocess
import signal
import gc
import requests
import selectors
import tempfile
from rich.errors import MarkupError
from rich.console import Console
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.model_loader.qaic import check_qpc_exists
from qaic_disagg.utils import (_id_type, remove_argument,
                               run_args_vllm_serve, run_args_proxy_server, set_nonblocking,
                               run_args_kvHandOff, get_next_color, create_rotating_writer,
                               display_progress_spinner_s, display_progress_spinner_e, is_device_ready,
                               check_device_qid_avail, _run_partitioner, post_dummy_request)
from qaic_disagg.proxy.utils import validate_and_get_host

from typing import (Any, Callable, Dict, List, Literal, Optional, Type, Tuple,
                    TypeVar, Union, cast, get_args, get_origin)
from vllm.worker.qaic_worker import QaicWorker
import json
from pathlib import Path

# Default extra compiler argument for Disaggregated Serving
default_encode_compile_args = {
    'aic-enable-depth-first': False
}

default_prefix_compile_args = {
    'allow-mxint8-mdp-io': True
}

default_decode_compile_args = {
    'split-retained-state-io': True,
    'allow-mxint8-mdp-io': True
}

default_env_vars ={
    'VLLM_ENGINE_ITERATION_TIMEOUT_S' : os.environ.get('VLLM_ENGINE_ITERATION_TIMEOUT_S','6000'),
    'VLLM_MDP_GENERATION_TIMEOUT_S' : os.environ.get('VLLM_MDP_GENERATION_TIMEOUT_S','6000'),
    'OMP_NUM_THREADS': os.environ.get("OMP_NUM_THREADS",'8'),
    'VLLM_QAIC_MAX_CPU_THREADS': os.environ.get('VLLM_QAIC_MAX_CPU_THREADS','8'),
    #'OMP_DISPLAY_ENV':'VERBOSE',
    #'OMP_DYNAMIC': 'TRUE',
    #'OMP_NESTED': 'TRUE',
}

QPC_DEACTIVATION_TIMEOUT = int(os.environ.get('QAIC_DISAGG_DEACTIVATION_TIMEOUT', 600))
QPC_ACTIVATION_TIMEOUT = int(os.environ.get('QAIC_DISAGG_ACTIVATION_TIMEOUT', 1200))
QPC_VERIFY_STATUS_TIMEOUT = int(os.environ.get('QAIC_DISAGG_VERIFY_STATUS_TIMEOUT', 300))

args_clean_up = ['router_policy',
                # 'skip_disagg_prefill_threshold',
                 'proxy_workers',
                 'kv_store_size',
                 'kv_handOff_port',
                 'encode_port',
                 'encode_device_group',
                 'encode_override_qaic_config',
                 'prefill_port',
                 'prefill_device_group',
                 'prefill_override_qaic_config',
                 'decode_port',
                 'decode_device_group',
                 'decode_override_qaic_config',
                 'encode_max_num_seqs',
                 'prefill_max_num_seqs',
                 'decode_max_num_seqs',
                 'verbose',
                 # Remove SSL related args from vLLM servers as the assumptions is that
                 # the Proxy server will be responsible for maintaining the SSL connection
                 # and connection from Proxy to vLLM server will be regular http
                 'ssl_keyfile',
                 'ssl_certfile',
                 'ssl_ca_certs',
                 'ssl_cert_reqs',
                 'enable_ssl_refresh',
                 'decode_speculative_config',
                 'post_dummy_request',
                 'prefill_max_seq_len_to_capture',
                 'decode_max_seq_len_to_capture',
                 'compile_only'
                 ]

pid = os.getpid()
# _print prefix
pprefix = f'[bold][dark_orange][JobManager | {pid}][/dark_orange][/bold]'
console = Console()
def _print(msg: str, prefix: str = pprefix) -> None:
    """ Overwrite _print function to add prefix """
    try:
        console.print(f"{prefix} {msg}")
    except MarkupError:
        console.print(f"{prefix} ", end="")
        console.print(msg, markup=False)

def validate_parsed_args(args: argparse.Namespace)-> argparse.Namespace:
    """
    Validate the parsed arguments.
    """

    # skip_prefill flag implies Decode has Prefill combined or separated
    # True ==> PD combined as D ==> E-PD disagg | False ==> P-D or E-P-D
    skip_prefill = True if (len(args.encode_port) != 0 and len(args.prefill_port) == 0) else False

    # Check if essential arguments are not provided
    if not args.decode_device_group:
        raise TypeError("--decode-device-group is required.")

    if not args.prefill_device_group and not args.encode_device_group:
        raise TypeError("--prefill-device-group or --encode-device-group are required.")

    if not args.decode_port:
        raise TypeError("--decode-port is required.")

    if not args.prefill_port and not args.encode_port:
        raise TypeError("--prefill-port or --encode-port are required.")

    # Check if number of instances provide for port ids and device groups are same
    if args.encode_device_group!=None and len(args.encode_device_group) != len(args.encode_port):
        raise TypeError("Number of encode port-ids and number of encode device group sets should match")

    if not skip_prefill and (len(args.prefill_device_group) != len(args.prefill_port)):
        raise TypeError("Number of prefill port-ids and number of prefill device group sets should match")

    if len(args.decode_device_group) != len(args.decode_port):
        raise TypeError("Number of decode port-ids and number of decode device group sets should match")

    if len(args.decode_device_group) == 0:
        raise TypeError("At least one decode device group set should be provided")

    if len(args.prefill_device_group) == 0 and len(args.encode_device_group) == 0:
        raise TypeError("At least one prefill or one encode device group set should be provided")

    # check if all port numbers are unique
    port_ids = [args.port] + args.decode_port + args.prefill_port + args.encode_port

    if len(port_ids) != len(set(port_ids)):
        raise TypeError(f"Port ids {port_ids} should be unique")

    for p in port_ids:
        if not 1 <= p <= 65535:
            raise TypeError(f"Port {p}must be between 1 and 65535.")

    if not args.compile_only:
        # Check if QIDs are unique for both prefill and decode instances
        _qids_inuse = set()
        _num_qid = 0

        if args.encode_port:
        # Check for encode QIDs are available
            _num_qid += check_device_qid_avail("encode", args.encode_device_group, _qids_inuse)

        # Check for prefill QIDs are available
        if not skip_prefill:
            _num_qid += check_device_qid_avail("prefill", args.prefill_device_group, _qids_inuse)

        # Check for decode QIDs are available
        _num_qid += check_device_qid_avail("decode", args.decode_device_group, _qids_inuse)

        if len(_qids_inuse) != _num_qid:
            raise TypeError(f"Duplicate QIDs found in encode {args.encode_device_group}, prefill {args.prefill_device_group} and decode {args.decode_device_group} device group")

    # check on override-qaic-configs
    if args.encode_device_group is not None and args.encode_override_qaic_config is not None:
        _keys = args.encode_override_qaic_config.keys()
        if 'mdp_load_partition_config' in _keys:
            raise TypeError(f"mdp_load_partition_config is not needed in encode override-qaic-configs!")
        if 'stages' in _keys:
            raise TypeError(f"stages is not required in encode override-qaic-configs!")
        if 'compile_only' in _keys:
            raise TypeError(f"compile_only is not needed in encode override-qaic-configs!")
        if 'prefill_only' in _keys:
            if args.encode_override_qaic_config['prefill_only'].lower() not in ('0',None,'None', ''):
                raise TypeError(f"prefill_only should be None in encode override-qaic-configs!")

    if not skip_prefill:
        # check on override-qaic-configs
        if args.prefill_override_qaic_config is not None and isinstance(args.prefill_override_qaic_config, dict):
            _keys = args.prefill_override_qaic_config.keys()
            if 'stages' in _keys:
                if len(args.prefill_device_group[0]) % int(args.prefill_override_qaic_config['stages']) != 0:
                    raise TypeError(f"Number of stages in prefill override-qaic-configs should be multiple of number of prefill device ids")
                if int(args.prefill_override_qaic_config['stages']) > len(args.prefill_device_group[0]):
                    raise TypeError(f"Number of stages in prefill override-qaic-configs should be less than or equal to number of prefill device ids")
            if 'compile_only' in _keys:
                raise TypeError(f"compile_only is not needed in prefill override-qaic-configs!")
            if 'prefill_only' in _keys:
                if args.prefill_override_qaic_config['prefill_only'].lower() not in ('true','1','none',None, ''):
                    raise TypeError(f"prefill_only should be True in prefill override-qaic-configs!")

    if args.decode_override_qaic_config is not None:
        #for key, value in args.decode_override_qaic_configs.items():
        _keys = args.decode_override_qaic_config.keys()
        if 'mdp_load_partition_config' in _keys:
            raise TypeError(f"mdp_load_partition_config is not needed in decode override-qaic-configs!")
        if 'stages' in _keys:
            raise TypeError(f"stages is not required in decode override-qaic-configs!")
        if 'compile_only' in _keys:
            raise TypeError(f"compile_only is not needed in decode override-qaic-configs!")
        if 'prefill_only' in _keys:
            if args.decode_override_qaic_config['prefill_only'].lower() not in ('false','0',None,'None', ''):
                raise TypeError(f"prefill_only should be False in prefill override-qaic-configs!")

    # check is 'skip-disagg-prefill-threshold' is less than CPL x prefill pipeline stages
    # if args.skip_disagg_prefill_threshold:
    #     if int(args.skip_disagg_prefill_threshold) >= (int(args.max_seq_len_to_capture) * len(args.prefill_device_group[0])):
    #         raise TypeError(f"skip_disagg_prefill_threshold should be less than CPL x prefill pipeline stages")

    # validate if host is valid and reachable
    args.host = validate_and_get_host(args.host)

def get_parser():
    """Get the parser for the CLI."""

    parser = FlexibleArgumentParser(description="QAIC Disaggregated serving using vLLM")
    parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Increase verbosity level. Can be specified multiple times."
        )

    parser = make_arg_parser(parser)
    remove_argument(parser, "--port")
    remove_argument(parser, "--device-group")
    remove_argument(parser, "--override-qaic-config")
    remove_argument(parser, "--tensor-parallel-size")
    remove_argument(parser, "--pipeline-parallel-size")
    remove_argument(parser, "--max-num-seqs")
    remove_argument(parser, "--max-seq-len-to-capture")

    qaic_disagg = parser.add_argument_group(
            title="Disaggregated",
            description="Option for using disaggregated serving",
        )

    qaic_disagg.add_argument(
            '--port',
            type=int,
            default=8000,
            help='Disaggregated proxy server port number')

    qaic_disagg.add_argument(
            "--encode-max-num-seqs",
            type=int,
            default=None,
            help="Maximum number of sequences to be processed in a single iteration"
        )

    qaic_disagg.add_argument(
            "--prefill-max-num-seqs",
            type=int,
            default=None,
            help="Maximum number of sequences to be processed in a single iteration"
        )

    qaic_disagg.add_argument(
            "--decode-max-num-seqs",
            type=int,
            required=True,
            help="Maximum number of sequences to be processed in a single iteration"
        )

    qaic_disagg.add_argument(
            "--encode-port",
            type=_id_type(),
            default=[],
            help="Port number for encode instances of vllm serve"
            "e.g. --encode-port 8001 or --encode-port 8000,8001,8002"
            " or --encode-port 8000:8002 or --encode-port 8000:8010:2"
            "If not specified, vllm will default port to empty list"
    )

    qaic_disagg.add_argument(
            "--prefill-max-seq-len-to-capture",
            type=int,
            default=256,
            help="Maximum sequence length that can be processed by a prefill instance in a single iteration"
        )

    qaic_disagg.add_argument(
            "--decode-max-seq-len-to-capture",
            type=int,
            default=None,
            help="Maximum sequence length that can be processed by a decode instance in a single iteration"
        )

    qaic_disagg.add_argument(
            "--prefill-port",
            type=_id_type(),
            default=[],
            help="Port number for prefill instances of vllm serve"
            "e.g. --prefill-port 8001 or --prefill-port 8000,8001,8002"
            " or --prefill-port 8000:8002 or --prefill-port 8000:8010:2"
            "If not specified, vllm will default port to empty list"
        )

    qaic_disagg.add_argument(
            "--decode-port",
            type=_id_type(),
            default=[8003],
            help="Port number for decode instances of vllm serve"
            "e.g. --decode-port 8001 or --decode-port 8000,8001,8002"
            " or --decode-port 8000:8002 or --prefill-port 8000:8010:2"
            "If not specified, vllm will use the default port 8003"
        )

    qaic_disagg.add_argument(
            '--encode-device-group',
            type=_id_type(),
            nargs='*',
            default=[],
            help=
            'Define qaic device ids in csv format (e.g., --device-id 0,1,2).')

    qaic_disagg.add_argument(
            '--prefill-device-group',
            type=_id_type(),
            nargs='*',
            default=[],
            help=
            'Define qaic device ids in csv format (e.g., --device-id 0,1,2).')

    qaic_disagg.add_argument(
            '--decode-device-group',
            type=_id_type(),
            nargs='*',
            default=[],
            help=
            'Define qaic device ids in csv format (e.g., --device-id 0,1,2).')

    qaic_disagg.add_argument(
        '--encode-override-qaic-config',
        type=lambda configs: {
            str(value[0]): value[1] if len(value) > 1 else True
            for value in
            (re.split(r'[:=]', config.strip()) for config in re.split(r'[ ]+', configs.strip()))
        },
        default=dict(),
        required=False,
        help="override or set qaic device configuration for encode instances.")

    qaic_disagg.add_argument(
        '--prefill-override-qaic-config',
        type=lambda configs: {
            str(value[0]): value[1] if len(value) > 1 else True
            for value in
            (re.split(r'[:=]', config.strip()) for config in re.split(r'[ ]+', configs.strip()))
        },
        default=dict(),
        required=False,
        help="override or set qaic device configuration for prefill instances.")

    qaic_disagg.add_argument(
        '--decode-override-qaic-config',
        type=lambda configs: {
            str(value[0]): value[1] if len(value) > 1 else True
            for value in
            (re.split(r'[:=]', config.strip()) for config in re.split(r'[ ]+', configs.strip()))
        },
        default=None,
        help="override or set qaic device configuration for decode instances.")

    qaic_disagg.add_argument(
        '--decode-speculative-config',
        type=lambda s: json.loads(s),
        default=dict(),
        required=False,
        help="Set speculative configuration for decode instances.")

    qaic_disagg.add_argument("--router-policy",
                "-r",
                type=str,
                choices=["round_robin", "least_outstanding"],
                default="round_robin",
                help="Scheduling policy to use for requests")

    qaic_disagg.add_argument("--kv-handOff-port",
            type=int,
            default=5656,
            help='ZMQ port number for kv Handoff control plane communication')

    qaic_disagg.add_argument("--kv-store-size",
                        "-S",
                        type=int,
                        default=64,
                        help="Size of the KV Cache store")

    qaic_disagg.add_argument("--proxy-workers",
                        "-w",
                        type=int,
                        default=None,
                        help="Number of uvicorn workers to use for the proxy server")

    # qaic_disagg.add_argument("--skip-disagg-prefill-threshold",
    #                     type=int,
    #                     default=None,
    #                     help="Skips disagg prefill if number of tokens is less than this threshold")

    qaic_disagg.add_argument("--post-dummy-request",
                            action="store_true",
                            help="Post dummy request upon server launch")

    qaic_disagg.add_argument("--compile-only",
                            action="store_true",
                            help="Perform compilation only and exit")

    return parser

class QaicDisaggJobManager():
    def __init__(self, args) -> None:
        self.args = args
        self.env = os.environ.copy()
        # check if environment variable VLLM_QAIC_QPC_PATH is set, then unset it
        if 'VLLM_QAIC_QPC_PATH' in self.env:
            del os.environ['VLLM_QAIC_QPC_PATH']
            _print("Warning: VLLM_QAIC_QPC_PATH environment variable is set. It will be unset, please use prefill-override-qaic-config or decode-override-qaic-config to provide qpc path")
        self.num_encode_instances = len(args.encode_port)
        self.num_prefill_instances = len(args.prefill_port)
        self.num_decode_instances = len(args.decode_port)
        self.skip_prefill = True if self.num_prefill_instances == 0 else False
        self.prefill_pp = len(args.prefill_device_group[0]) if not self.skip_prefill else None
        self.host = self.args.host
        self.kv_rank = 0
        self.proc_handle = {
                'encode': [],
                'prefill': [],
                'decode': [],
                'proxy' : [],
                'kvHandOff': []
            }
        self.instance_drop_regex = re.compile("Due to higher number of exceptions ")

        self.file_handles = dict()
        self.file_buff = dict()
        self.sel = selectors.DefaultSelector()
        self.shutdown_done = False
        self.started_exit = False
        self.force_exit = False
        self.all_gracefull_exit = True
        self.post_dummy = self.args.post_dummy_request
        # setup signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def __del__(self):
        if not self.started_exit:
            self.shutdown()

    def shutdown(self):
        """
        Shutdown the vLLM serve process for a specific instance.
        """
        if self.shutdown_done:
            return

        self.sync_prints()
        _print("Gracefully shutting down all child processes...")
        start_time = time.time()

        def calculate_remaining_shutdown_time():
            return max(0, QPC_DEACTIVATION_TIMEOUT - (time.time() - start_time))

        def _safe_kill_pg(pid: int, sig: int):
            """
            Safely attempt to send a signal to a process group.

            Args:
                pid: Process group ID.
                signal: Signal to send (e.g., signal.SIGTERM).
            """
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                # Process group may already have exited
                _print(f"Process {pid} is already terminated")
            except Exception as e:
                _print(f"Error terminating process {pid}; it may not be fully terminated: {e}")

        def shutdown_processes(process: List[str] | str):
            try:
                if isinstance(process, str):
                    process = [process]

                # Send SIGINT to process only if force exit is not in progress
                if not self.force_exit:
                    for name in process:
                        for p in self.proc_handle.get(name, []):
                            _print(f"Forwarding signal {signal.SIGINT} to {name} process {p.pid}")
                            try:
                                if p.poll() is None:
                                    os.killpg(p.pid, signal.SIGINT)
                                else:
                                    # The process is already terminated, send SIGKILL to
                                    # process group to kill any lingering child processes
                                    os.killpg(p.pid, signal.SIGKILL)
                            except ProcessLookupError: # Process may already be terminated
                                _print(f"{name} process {p.pid} is already terminated")
                            except Exception as e:
                                _print(f"Error terminating {name} process {p.pid}; it may not be fully terminated: {e}")

                # Poll till shutdown is complete or timeout is triggered
                for name in process:
                    for p in self.proc_handle.get(name, []):
                        if not self.force_exit:
                            _print(f"Waiting for shutdown of {name} process {p.pid}")
                            while p.poll() is None:
                                if self.args.verbose < 2:
                                    display_progress_spinner_s()

                                self.sync_prints()
                                remaining_time = calculate_remaining_shutdown_time()

                                if self.force_exit:
                                    _print(f"Killing {name} process {p.pid} immediately.")
                                    _safe_kill_pg(p.pid, signal.SIGKILL)
                                    break
                                elif remaining_time <= 0:
                                    _print(f"Shutdown timeout ({QPC_DEACTIVATION_TIMEOUT} seconds) exceeded")
                                    self.force_exit = True
                                    self.all_gracefull_exit = False

                                if self.args.verbose < 2:
                                    display_progress_spinner_e()

                                time.sleep(0.1)
                        elif p.poll() is None: # if force exit then kill all remaining processes
                            _print(f"Killing {name} process {p.pid} immediately.")
                            _safe_kill_pg(p.pid, signal.SIGKILL)
            except Exception as e:
                _print(f"[yellow] Error while {process} process shutdown: {e}[/yellow]")

        # Step 1: Shutdown vLLM instances
        shutdown_processes(["encode", "prefill", "decode"])
        self.sync_prints()

        # Step 2: Shutdown kvhandoff server
        if not self.skip_prefill:
            shutdown_processes("kvHandOff")
            self.sync_prints()

        # Step 3: Shutdown proxy server
        shutdown_processes("proxy")
        self.sync_prints()

        # Step 4: Unregister selectors
        if len(self.sel.get_map()) > 0:
            for key, _ in self.sel.select():
                fd = key.fd
                self.sel.unregister(fd)
        for _, handle in self.file_handles.items():
            handle.close()
        self.sel.close()

        # Step 5: Verify device status
        _print("Verifying device health")
        self.verify_devices()

        if not self.all_gracefull_exit:
            _print(f"[yellow]One or more processes failed to shut down gracefully. Please ensure that all assigned ports and devices have been released.")

        _print("All child processes terminated.")
        self.shutdown_done = True
        self.proc_handle = dict()
        self.sel = dict()
        self.file_handles = dict()

    def verify_devices(self):
        """
        Verify if used devices are available within a timeout.
        """
        sleep_interval = 5  # seconds
        start_time = time.time()

        devices = [qid for qid_group in self.args.prefill_device_group + self.args.decode_device_group + self.args.encode_device_group for qid in qid_group]
        pending_devices = set(devices)

        while time.time() - start_time < QPC_VERIFY_STATUS_TIMEOUT:
            validated_devices = {qid for qid in pending_devices if is_device_ready(qid)}
            pending_devices -= validated_devices

            if not pending_devices:
                return  # All devices are available
            time.sleep(sleep_interval)

        # Final check after timeout
        validated_devices = {qid for qid in pending_devices if is_device_ready(qid)}
        pending_devices -= validated_devices

        if pending_devices:
           self.all_gracefull_exit = False
           _print(f"[yellow]Device id(s) {', '.join(map(str, sorted(pending_devices)))} are not in available state after {QPC_VERIFY_STATUS_TIMEOUT} seconds timeout [/yellow]")

    def signal_handler(self, signum, frame):
        if self.started_exit and signum == signal.SIGINT:
            _print(f"Please wait... Shutdown in progress with a timeout of {QPC_DEACTIVATION_TIMEOUT} seconds.")
        else:
            _print(f"Signal {signum} received. Terminating all child processes.")
            self.started_exit = True
            self.shutdown()
            sys.exit(0)

    def get_args(self, instance_id: int, instType:str) -> argparse.Namespace:
        """
        Get the arguments for a specific instance.
        """

        args = copy.deepcopy(self.args)

        args.port = getattr(args, f"{instType}_port")[instance_id]
        args.device_group = getattr(args, f'{instType}_device_group')[instance_id]
        args.override_qaic_config = getattr(args, f'{instType}_override_qaic_config', None)

        if not args.override_qaic_config:
            args.override_qaic_config = {}

        extra_comp_args = None
        if instType == 'encode':
            args.max_num_seqs = args.encode_max_num_seqs
            extra_comp_args = default_encode_compile_args
        elif instType =='prefill':
            args.override_qaic_config['stages'] = len(args.prefill_device_group[instance_id])
            args.override_qaic_config['prefill_only'] = True
            args.max_num_seqs = args.prefill_max_num_seqs
            args.max_seq_len_to_capture = args.prefill_max_seq_len_to_capture
            extra_comp_args = default_prefix_compile_args
        elif instType =='decode':
            args.max_num_seqs = args.decode_max_num_seqs
            args.max_seq_len_to_capture = args.decode_max_seq_len_to_capture if args.decode_max_seq_len_to_capture else args.prefill_max_seq_len_to_capture
            # if args.skip_disagg_prefill_threshold:
            #     args.override_qaic_config['prefill_only'] = None
            # else:
            #     args.override_qaic_config['prefill_only'] = False
            extra_comp_args = default_decode_compile_args.copy()
            if not self.skip_prefill:
                args.override_qaic_config['prefill_only'] = False
            else:
                extra_comp_args.pop('split-retained-state-io')

        for key in extra_comp_args.keys():
            if key not in args.override_qaic_config:
                args.override_qaic_config[key] = extra_comp_args[key]

        if not self.skip_prefill:
            args.tensor_parallel_size = 1
            args.pipeline_parallel_size = 1

        # Set speculative config args
        if instType == 'decode':
            if args.decode_speculative_config is not None:
                args.speculative_config = args.decode_speculative_config

        for _arg in args_clean_up:
            if hasattr(args, _arg) or 'speculative' in _arg:
                delattr(args, _arg)
        return args

    def compile_qpcs(self, instType:str, mdp_dump_partition_config_path: Optional[str] = None) -> None:
        args = self.get_args(0, instType)
        compile_type = "compile"
        if instType == "prefill" and mdp_dump_partition_config_path:
            args.override_qaic_config['mdp_dump_partition_config'] = mdp_dump_partition_config_path
            compile_type = "mdp_gen"
        compilation_sucessfull = False
        if args.override_qaic_config and 'qpc_path' in args.override_qaic_config:
            qpc_path = args.override_qaic_config['qpc_path']
            if qpc_path and check_qpc_exists(qpc_path):
                _print(f"Skipping compilation, found QPC path for {instType} instance in override_qaic_config")
                return
            else:
                _print(f"Warning: QPC path {qpc_path} provided not correct for {instType}")
                del args.override_qaic_config['qpc_path']

        args.override_qaic_config['compile_only'] = True # exit after compilation
        fd = None
        color= get_next_color()
        try:
            prefill_filename = f"qaic_vllm_{instType}_{compile_type}.log"
            filehandle = open(prefill_filename, 'w', buffering=1) if self.args.verbose > 1 else create_rotating_writer(prefill_filename)
            proc = subprocess.Popen(
                run_args_vllm_serve(args, instType, self.skip_prefill,
                                    self.kv_rank, str(self.args.host), self.args.kv_handOff_port,
                                    verbose=self.args.verbose),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            prefix = f'[bold][{color}]\[{instType}_{compile_type} | {proc.pid}][/{color}][/bold]'
            _print(f"Started compilation for {instType} qpc with pid {proc.pid}, log {prefill_filename}")
            self.register_proc(proc, instType, prefix, filehandle)
            while proc.poll() is None:
                if self.args.verbose < 2:
                    display_progress_spinner_s()
                # Dump prints
                for key, _ in self.sel.select(timeout=1):
                    _, _, _, writer = key.data
                    fd = key.fd
                    try:
                        chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    except Exception:
                        chunk = ""
                        pass
                    if chunk:
                        self.file_buff[fd] += chunk
                    while '\n' in self.file_buff[fd]:
                        line, self.file_buff[fd] = self.file_buff[fd].split('\n', 1)
                        if self.args.verbose > 1 :
                            writer.write(line+'\n')
                        else:
                            writer.stream.write(line+'\n')
                            writer.flush()

                        if self.args.verbose > 2:
                            _print(line, prefix=prefix)

                        if 'Using qpc' in line:
                            compilation_sucessfull = True
                            if ':-' in line:
                                qpc_path = line.strip().split(':-')[-1]
                                _print(f"QPC sucessfully compiled for {instType}, qpc_path={qpc_path}")
                                if instType == "prefill" and mdp_dump_partition_config_path:
                                    continue
                                else:
                                    _print(f"Updating {instType}_override_qaic_config with qpc_path={qpc_path}")
                                    if getattr(self.args, f"{instType}_override_qaic_config") is None:
                                        setattr(self.args, f"{instType}_override_qaic_config", {"qpc_path": qpc_path})
                                    else:
                                        getattr(self.args, f"{instType}_override_qaic_config")['qpc_path'] = qpc_path
                            else:
                                _print(f"QPC sucessfully compiled for {instType}")
                time.sleep(1)
                if self.args.verbose < 2:
                    display_progress_spinner_e()
        except Exception as e:
            _print(f"Exception is passed still gracefull exit!! {e}")
            pass
        finally:
            #cleanups
            self.sync_prints()
            if filehandle:
                filehandle.close()
            self.proc_handle[instType] = []
            if fd:
                if fd in self.file_buff:
                    del self.file_buff[fd]
                if fd in self.file_handles:
                    del self.file_handles[fd]
                if len(self.sel.get_map()) > 0:
                    self.sel.unregister(fd)
        self.kv_rank = 0
        if not compilation_sucessfull:
            raise ValueError(f"Compilation failed for {instType} instance")

    def wait_for_vllm_instance_to_up(self):
        start = time.perf_counter()
        nb_qpc_loaded = 0
        nb_vllm_inst = len(self.args.prefill_port) + len(self.args.decode_port) + len(self.args.encode_port)
        abort_needed = False
        nb_qpc_loading_ongoing = set()
        st_qpc_load_regex = re.compile(r"Loading QPC..")
        end_qpc_load_regex = re.compile(r"Successfully loaded QPC")
        uvicorn_ready_regex = re.compile(r"Application startup complete")

        try:
            while  nb_qpc_loaded != nb_vllm_inst:
                if self.args.verbose < 2:
                    display_progress_spinner_s()
                for key, procL in self.proc_handle.items():
                    for proc in procL:
                        if not abort_needed and proc.poll() is not None:
                            abort_needed = True
                            _print(f"{key} instance with pid {proc.pid} failed, will abort shortly")

                for proc in list(nb_qpc_loading_ongoing):
                    if proc.poll() is not None:
                        abort_needed = True
                        nb_qpc_loading_ongoing.remove(proc)

                # Dump prints
                for key, _ in self.sel.select(timeout=1):
                    instType, proc, prefix, writer = key.data
                    fd = key.fd
                    try:
                        chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
                    except Exception:
                        chunk = ""
                        pass
                    if chunk:
                        self.file_buff[fd] += chunk
                    while '\n' in self.file_buff[fd]:
                        line, self.file_buff[fd] = self.file_buff[fd].split('\n', 1)
                        if self.args.verbose > 1 :
                            writer.write(line+'\n')
                        else:
                            writer.stream.write(line+'\n')
                            writer.flush()

                        if self.args.verbose > 2:
                            _print(line, prefix=prefix)

                        if st_qpc_load_regex.search(line):
                            _print(f"Loading QPC for {instType} instance {prefix}")
                            nb_qpc_loading_ongoing.add(proc)

                        if end_qpc_load_regex.search(line):
                            _print(f"QPC loaded sucessfully for {instType} instance {prefix}")
                            nb_qpc_loading_ongoing.remove(proc)

                        if uvicorn_ready_regex.search(line):
                            _print(f"API server started for {instType} instance {prefix}")
                            nb_qpc_loaded += 1

                if abort_needed and len(nb_qpc_loading_ongoing) == 0:
                    #self.shutdown()
                    raise RuntimeError(f"vLLM instances failed check the log files")

                if (time.perf_counter() - start) > QPC_ACTIVATION_TIMEOUT:
                    raise TimeoutError(f"vLLM Servers failed to start in {QPC_ACTIVATION_TIMEOUT} seconds")

                if self.args.verbose < 2:
                    display_progress_spinner_e()

        except Exception as e:
            raise RuntimeError(f"{e}")

    def sync_prints(self):
        for key, _ in self.sel.select(timeout=1):
            instType, proc, prefix, writer = key.data
            fd = key.fd
            disable_print = False
            try:
                _chunk = os.read(fd, 1048576)  # 1 MB buffer
                chunk = _chunk.decode()
            except UnicodeDecodeError:
                chunk = _chunk.decode("utf-8", errors="replace")
                disable_print = True
            except Exception:
                chunk  = ""
            if chunk:
                self.file_buff[fd] += chunk
            while '\n' in self.file_buff[fd]:
                line, self.file_buff[fd] = self.file_buff[fd].split('\n', 1)
                if writer:
                    if self.args.verbose > 1 :
                        writer.write(line+'\n')
                    else:
                        if writer.stream:
                            writer.stream.write(line+'\n')
                            writer.flush()
                if self.args.verbose > 2 and not disable_print:
                    _print(line, prefix=prefix)
                if self.instance_drop_regex.search(line):
                    _print("[bold][red1] vLLM Instance dropped !! [/red1][/bold]")

    def monitor(self):
        """
        Monitor the instances and restart them if they fail.
        """
        _essential_proc_handle = []
        _optional_proc_handle = []
        try:
            self.sync_prints()
            for k,v in self.proc_handle.items():
                for proc in v:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{k} service with pid {proc.pid} failed")
                if len(v) != 0:
                    if k not in ['decode', 'prefill', 'encode']:
                        _essential_proc_handle.append(v[0])
                    else:
                        _essential_proc_handle.append(v[0])
                        _optional_proc_handle.extend(v[1:])
                else:
                    continue
        except Exception as e:
            raise RuntimeError(f"Monitoring failed with error: {e}")
        time.sleep(1)
        # Everything should be up by now
        url = f"http://{self.host}:{self.args.port}/v1/chat/completions"
        # TODO: The dummy request will fail over an SSL connection. Skipping for now. Need to fix.
        if self.args.ssl_keyfile:
           _print(f"[green]Qaic Disaagregated serving end point[/green] [red]{url}[/red] [green]is up now [/green]")
        elif self.post_dummy:
                try:
                    self.sync_prints()
                    if not self.skip_prefill:
                        response = post_dummy_request(url, "text", self.args.model)
                    else:
                        if "height" not in self.args.encode_override_qaic_config or "width" not in self.args.encode_override_qaic_config:
                            response = post_dummy_request(url, "image+text", self.args.model)
                        else:
                            height = int(self.args.encode_override_qaic_config["height"])
                            width = int(self.args.encode_override_qaic_config["width"])
                            response = post_dummy_request(url, "image+text", self.args.model, height, width)

                    if response.status_code == 200:
                        _print(f"[green]Qaic Disaagregated serving end point[/green] [red]{url}[/red] [green]is up now [/green]")
                        pass
                except Exception as e:
                    _print(e)
                    #_print(f"Qaic Disaagregated serving end point [red]{url}[/red] is not up yet. Please try again.")
                    pass
        _print(f"Monitoring child processes in background.")
        _print(f"Press [yellow]Ctl-C[/yellow] once to shutdown all services.")
        # Start monitoring the instances
        while True:
            try:
                self.sync_prints()
                for proc in _essential_proc_handle:
                    if proc.poll() is not None:
                        self.all_gracefull_exit = False
                        raise RuntimeError(f"Essential service with pid {proc.pid} failed")
                for proc in _optional_proc_handle:
                    if proc.poll() is not None:
                        _print(f"Warning: Service with pid {proc.pid} failed")
                # time.sleep(5)
            except Exception as e:
                raise RuntimeError(f"Monitoring failed with error: {e}")

    def register_proc(self, proc, instType, prefix, filehandle):
        for handle in [proc.stdout, proc.stderr]:
            fd = set_nonblocking(handle)
            assert fd not in self.file_buff
            self.sel.register(fd, selectors.EVENT_READ, (instType, proc, prefix, filehandle))
            self.file_buff[fd] = ""
            self.file_handles[fd] = filehandle
        self.proc_handle[instType].append(proc)
        self.kv_rank += 1

    def run(self)-> None:
        """
        Run the main function.
        """
        self.env.update(default_env_vars)

        # Check for common mistakes
        if 'HF_TOKEN' not in self.env:
            _print("[bold][red1]Warning: HF_TOKEN environment variable not set. Please set it to your Hugging Face authentication token[/red1][/bold]")
        if 'QEFF_HOME' not in self.env:
            _print("Error: QEFF_HOME environment variable not set. Please set it to your QEfficient cache directory")
            raise AttributeError("QEFF_HOME environment variable not set. Please set it to your QEfficient cache directory")
        if 'HF_HOME' not in self.env:
            _print("Error: HF_HOME environment variable not set. Please set it to your Hugging Face cache directory")
            raise AttributeError("HF_HOME environment variable not set. Please set it to your Hugging Face cache directory")

        if not self.skip_prefill:
            _print(f"Starting [bold][indian_red1]{len(self.args.prefill_port)}P{len(self.args.decode_port)}D[/indian_red1][/bold] configuration..")
            # Generate mdp file.
            args = self.get_args(0, 'prefill')
            if ('mdp_load_partition_config' not in args.override_qaic_config and
                'stages' in args.override_qaic_config and
                int(args.override_qaic_config['stages']) > 1 and
                'qpc_path' not in self.args.prefill_override_qaic_config):
                _print(f"Generating MDP file for prefill...")
                mdp_dump_partition_config = os.path.join(os.getcwd(), "model_partitioned_prefill_mdp.json")
                out_dir = Path(mdp_dump_partition_config).absolute().parent
                temp_mdp_dump_json_path = str(out_dir / "tmp_mdp_dump_partition.json")

                # Call compile_qpcs to essentially get the tmp mdp JSON file
                self.compile_qpcs('prefill', temp_mdp_dump_json_path)

                if not os.path.exists(temp_mdp_dump_json_path):
                    raise RuntimeError("Failed to generate temporary MDP file")

                # Run partitioner on the tmp mdp JSON file
                _run_partitioner(
                args,
                temp_mdp_dump_json_path,
                mdp_dump_partition_config)

                if not os.path.exists(mdp_dump_partition_config):
                    raise RuntimeError("Failed to generate MDP file for prefill")

                # Cleanup tmp files
                #if os.path.exists(temp_mdp_dump_json_path):
                #    os.remove(temp_mdp_dump_json_path)
                self.args.prefill_override_qaic_config['mdp_load_partition_config'] = mdp_dump_partition_config
                _print(f"Generated MDP file for prefill instances at: {mdp_dump_partition_config}")

        else:
            _print(f"Starting [bold][indian_red1]{len(self.args.encode_port)}E{len(self.args.decode_port)}D[/indian_red1][/bold] configuration..")

        # Compile qpcs for encode, prefill and decode instances
        if self.num_encode_instances > 0:
            self.compile_qpcs('encode')
        if not self.skip_prefill:
            self.compile_qpcs('prefill')
        self.compile_qpcs('decode')

        # Exit if compile-only is true
        if self.args.compile_only:
            _print("Exiting after compilation.")
            return

        cfg_path = tempfile.gettempdir() + "/qti_qaic/qaic_disagg/usage/"

        # Start vllm instances for encode
        for instance_id in range(len(self.args.encode_port)):
            self.env['VLLM_CONFIG_ROOT'] = cfg_path + f'encode_{instance_id}'
            args = self.get_args(instance_id, 'encode')
            color = get_next_color()
            encode_filename = f"qaic_vllm_encode_{instance_id}.log"
            filehandle = open(encode_filename, 'w', buffering=1) if self.args.verbose > 1 else create_rotating_writer(encode_filename)
            proc = subprocess.Popen(
                run_args_vllm_serve(args, 'encode', self.skip_prefill, verbose=self.args.verbose),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=self.env,
                start_new_session=True
            )
            _print(f"Started encode{instance_id} instance with pid [{color}]{proc.pid}[/{color}], port [{color}]{args.port}[/{color}], log {encode_filename}")
            prefix = f'[bold][{color}][Encode{instance_id}_p{args.port} | {proc.pid}][/{color}][/bold]'
            self.register_proc(proc, 'encode', prefix, filehandle)

        # Start vllm instances for prefill
        if not self.skip_prefill:
            for instance_id in range(len(self.args.prefill_port)):
                self.env['VLLM_CONFIG_ROOT'] = cfg_path + f'prefill_{instance_id}'
                args = self.get_args(instance_id, 'prefill')
                color = get_next_color()
                prefill_filename = f"qaic_vllm_prefill_{instance_id}.log"
                filehandle = open(prefill_filename, 'w', buffering=1) if self.args.verbose > 1 else create_rotating_writer(prefill_filename)
                proc = subprocess.Popen(
                    run_args_vllm_serve(args,'prefill', self.skip_prefill,
                                        self.kv_rank, str(self.args.host), self.args.kv_handOff_port,
                                        verbose=self.args.verbose),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=self.env,
                    start_new_session=True
                )
                _print(f"Started prefill{instance_id} instance with pid [{color}]{proc.pid}[/{color}], port [{color}]{args.port}[/{color}], log {prefill_filename}")
                prefix = f'[bold][{color}][Prefill{instance_id}_p{args.port} | {proc.pid}][/{color}][/bold]'
                self.register_proc(proc, 'prefill', prefix, filehandle)

        #start vllm instances for decode
        for instance_id in range(len(self.args.decode_port)):
            self.env['VLLM_CONFIG_ROOT'] = cfg_path + f'decode_{instance_id}'
            args = self.get_args(instance_id, 'decode')
            color = get_next_color()
            decode_fname = f"qaic_vllm_decode_{instance_id}.log"
            filehandle = open(decode_fname, 'w', buffering=1) if self.args.verbose > 1 else create_rotating_writer(decode_fname)
            proc = subprocess.Popen(
                    run_args_vllm_serve(args, 'decode', self.skip_prefill,
                                        self.kv_rank, str(self.args.host), self.args.kv_handOff_port,
                                        #skip_disagg_prefill_threshold=self.args.skip_disagg_prefill_threshold,
                                        verbose=self.args.verbose),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    env=self.env,
                    start_new_session=True
                )
            _print(f"Started decode{instance_id} instance with pid [{color}]{proc.pid}[/{color}], port [{color}]{args.port}[/{color}], log {decode_fname}")
            prefix = f'[bold][{color}][Decode{instance_id}_p{args.port} | {proc.pid}][/{color}][/bold]'
            self.register_proc(proc, 'decode', prefix, filehandle)

        if not self.skip_prefill:
            color = get_next_color()
            kvhandoff_filename = f"qaic_kvhandoff.log"
            filehandle = open(kvhandoff_filename, 'w', buffering=1) if self.args.verbose > 1 else create_rotating_writer(kvhandoff_filename)
            proc = subprocess.Popen(
                    run_args_kvHandOff(self.args),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True
                )
            _print(f"Started Qaic KvHandOff service with pid [{color}]{proc.pid}[/{color}], log {kvhandoff_filename}")
            prefix = f'[bold][{color}][KVHandOff_p{self.args.kv_handOff_port} | {proc.pid}][/{color}][/bold]'
            self.register_proc(proc, 'kvHandOff', prefix, filehandle)

        # # Wait for all vllm instances to UP
        _print(f"Waiting for vllm instances to start, timeout set for {QPC_ACTIVATION_TIMEOUT} seconds...")
        self.wait_for_vllm_instance_to_up()
        time.sleep(15)
        #start vllm instances for decode
        color = get_next_color()
        proxy_filename = f"qaic_disagg_proxy.log"
        filehandle = open(proxy_filename, 'w', buffering=1) if self.args.verbose > 1 else create_rotating_writer(proxy_filename)
        env=None
        # if the user supplies api_key we set it as VLLM_API_KEY so that proxy server
        # can validate the instances that are registered to it.
        if self.args.api_key:
            env = os.environ.copy()
            env["VLLM_API_KEY"] = str(self.args.api_key)
        proc = subprocess.Popen(
                run_args_proxy_server(self.args, self.skip_prefill),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True
            )
        if not self.skip_prefill:
            _print(f"Started proxy service to manage prefill and decode instances with pid [{color}]{proc.pid}[/{color}], log {proxy_filename}")
        else:
            _print(f"Started proxy service to manage encode and decode instances with pid [{color}]{proc.pid}[/{color}], log {proxy_filename}")
        prefix = f'[bold][{color}][Proxy_p{self.args.port} | {proc.pid}][/{color}][/bold]'
        self.register_proc(proc, 'proxy', prefix, filehandle)
        time.sleep(20)
        # Wait for proxy services to be up
        self.monitor()

def main():
    if not current_platform.is_qaic():
        raise RuntimeError("This script is only supported for vLLM QAIC backend!")
    parser: FlexibleArgumentParser = get_parser()
    args = parser.parse_args()
    validate_parsed_args(args)
    disagg = QaicDisaggJobManager(args)
    disagg.run()

if __name__ == "__main__":
    main()