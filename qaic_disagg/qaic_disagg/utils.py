# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
import argparse
from collections.abc import Iterator
from typing import (Any, Callable, Optional, TypeVar, Union, cast, List)
import fcntl
import os
import zmq
import itertools
import importlib.util
import json
import sys
import concurrent.futures
import functools
import threading
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from vllm.utils import is_valid_ipv6_address, make_zmq_socket
from PIL import Image
import base64
from io import BytesIO
import requests
import subprocess
from transformers import AutoConfig

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint = Union[type[Any], object]
TypeHintT = Union[type[T], object]

# Color used for printing messages to the console.
# Applicable to 'rich' library
colors = [
    "dark_slate_gray2", "green", "yellow", "blue", "magenta",
    "cyan","bright_green", "bright_yellow", "bright_blue",
    "bright_magenta", "bright_cyan", "bright_white", "grey",
    "purple", "orange1", "deep_pink1", "chartreuse1", "deep_sky_blue1",
    "light_goldenrod1", "bright_red", "medium_purple", "turquoise2", 
    "gold1", "red"
]

color_idx = 0
def get_next_color() -> str:
    global color_idx
    if color_idx == len(colors)-1:
        color_idx = 0
    color = colors[color_idx]
    color_idx += 1
    return color

def create_rotating_writer(fname, max_bytes=1024 * 1024, backup_count=3):
    handler = RotatingFileHandler(fname, mode='w', maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(None)
    return handler

def set_nonblocking(fileobj):
    fd = fileobj.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    return fd

spinner = itertools.cycle(['|', '/', '-', '\\'])

def display_progress_spinner_s():
    sys.stdout.write(next(spinner))
    sys.stdout.flush()

def display_progress_spinner_e():
    sys.stdout.write('\b')
    sys.stdout.flush()

def _id_type() -> Callable[[str], Optional[T]]:
    def _optional_type(val: str) -> Optional[T]:
        if val == "" or val == "None":
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to port ids.")

        try:
            ids = []
            _set = set()
            _num_ids = 0
            val = val.strip().split(",")
            for v in val:
                if ':' in v:
                    parts = list(map(int, v.split(':')))
                    _ids = list(range(*parts))
                elif '..' in v:
                    parts = list(map(int, v.split('..')))
                    _ids = list(range(parts[0],parts[1]+1))
                else:
                    _ids = int(v)
                _num_ids+= len(_ids) if isinstance(_ids, list) else 1
                _set.update(_ids) if isinstance(_ids, list) else _set.add(_ids)
                ids.extend(_ids) if isinstance(_ids, list) else ids.append(_ids)
            if _num_ids != len(_set):
                raise argparse.ArgumentTypeError(
                    f"Set of values {val} not unique.")

            return cast(T, ids)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to ids.") from e

    return _optional_type

def remove_argument(parser, option_string):
    action = parser._option_string_actions.pop(option_string, None)
    if not action:
        return

    try:
        parser._actions.remove(action)
        for option in action.option_strings:
            parser._option_string_actions.pop(option, None)

        # Remove from _action_groups (used for help display)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)
    except:
        raise ValueError(f"Option {option_string} not found in parser.")


def update_help(parser, option_string, new_help):
    action = parser._option_string_actions.get(option_string, None)
    if action:
        action.help = new_help

def run_args_vllm_serve(args: argparse.Namespace, instType:str, skip_prefill: bool,
                        kv_rank:Optional[int]=None, kv_ip:Optional[str]=None, kv_port:Optional[int]=None,
                        skip_disagg_prefill_threshold:Optional[int]=None, verbose:Optional[int]=1)-> None:
    """
    Start the vllm serve process for a specific instance.
    """

    # Convert Namespace to list of CLI arguments
    arg_list = []
    model_name = args.model
    for key, value in vars(args).items():
        if not value:
            continue

        if key in ['config', 'model']:
            continue

        if isinstance(value, list):
            if len(value) ==0:
                continue
            value = ','.join(map(str, value))

        if isinstance(value, dict):
            if len(value) == 0:
                continue
            if key == 'override_qaic_config':
                _value = ""
                for k,v in value.items():
                    if isinstance(v, list):
                        v = ','.join(map(str, v))
                    else:
                        v = str(v)
                    _value += f"{k}={v} "
                value = _value.strip()
            else:
                value = json.dumps(value)

        key = key.replace('_', '-')
        if value in ['auto', '', '*']:
            continue
        if isinstance(value, bool) and value:
            arg_list.append(f"--{key}")
        elif isinstance(value, bool) and not value:
            continue
        else:
            arg_list.append(f"--{key}")
            arg_list.append(f"{value}")

    if verbose < 3:
        if '--disable-log-stats' not in arg_list:
            arg_list.append('--disable-log-stats')
    if verbose < 4:
        if '--disable-log-requests' not in arg_list:
            arg_list.append('--disable-log-requests')

    if not skip_prefill:
        # Adding KV transfer config
        arg_list.append('--kv-transfer-config')
        kv_config = "{\"kv_connector\":\"QaicConnector\", \"kv_role\":"

        if instType == 'prefill':
            kv_config += f"\"kv_producer\""
        elif skip_disagg_prefill_threshold:
            kv_config += f"\"kv_both\""
        else:
            kv_config += f"\"kv_consumer\""
        kv_config += f", \"kv_rank\": {kv_rank}, \"kv_ip\": \"{kv_ip}\", \"kv_port\": {kv_port} "
        kv_config += "}"
        arg_list.append(kv_config)

    if instType == 'encode':
        arg_list.append("--task")
        arg_list.append("embed")
        arg_list.append("--disable-mm-preprocessor-cache")

    return ['vllm', 'serve', f'{model_name}'] + arg_list

def run_args_proxy_server(args: argparse.Namespace, skip_prefill: bool)-> None:
    """
    Start the proxy server process for a specific instance.
    """
    # Convert Namespace to list of CLI arguments
    args_list = [sys.executable,
                 '-m',
                 'qaic_disagg.proxy.app',
                 '--model',
                 f'{args.model}',
                 '--router-policy',
                 f'{args.router_policy}',
                 '--host',
                 f'{args.host}',
                 '--port',
                 f'{args.port}']

    if args.proxy_workers:
        args_list.append('--workers')
        args_list.append(f'{args.proxy_workers}')

    # Add SSL arguments if they are provided
    if args.ssl_keyfile:
        args_list += ['--ssl-keyfile', args.ssl_keyfile]
    if args.ssl_certfile:
        args_list += ['--ssl-certfile', args.ssl_certfile]
    if args.ssl_ca_certs:
        args_list += ['--ssl-ca-certs', args.ssl_ca_certs]
    if args.ssl_cert_reqs is not None:
        args_list += ['--ssl-cert-reqs', str(args.ssl_cert_reqs)]

    # Add uvicorn log arguments
    if args.disable_uvicorn_access_log:
        args_list += ['--disable-uvicorn-access-log']
    if args.uvicorn_log_level:
        args_list += ['--uvicorn-log-level', args.uvicorn_log_level]

    # if args.skip_disagg_prefill_threshold:
    #     args_list.append('--skip-disagg-prefill-threshold')
    #     args_list.append(f'{args.skip_disagg_prefill_threshold}')

    host = args.host
    if is_valid_ipv6_address(host):
        host = '[' + host + ']'

    # Only add --encode flag if there are encode ports
    if args.encode_port and len(args.encode_port) > 0:
        args_list.append('--encode')
        for p in args.encode_port:
            args_list.append(f'{host}:{p}')

    if not skip_prefill:
        args_list.append('--prefill')
        for p in args.prefill_port:
            args_list.append(f'{host}:{p}')

    args_list.append('--decode')
    for p in args.decode_port:
        args_list.append(f'{host}:{p}')

    # for vision encode
    if args.encode_device_group:
        if 'height' in args.encode_override_qaic_config:
            args_list.append('--height')
            args_list.append(f'{args.encode_override_qaic_config["height"]}')
        if 'width' in args.encode_override_qaic_config:
            args_list.append('--width')
            args_list.append(f'{args.encode_override_qaic_config["width"]}')
        args_list.append('--num-frames')
        args_list.append(f'{args.limit_mm_per_prompt["image"]}')

    return args_list

def run_args_kvHandOff(args: argparse.Namespace)-> None:
    """
    Start the decode process for a specific instance.
    """
    # Convert Namespace to list of CLI arguments
    args_list = [sys.executable,
                 '-m',
                 'qaic_disagg.kv_handoff.server',
                 '--host',
                 f'{args.host}',
                 '--port',
                 f'{args.kv_handOff_port}',
                 '--size',
                 f'{args.kv_store_size}']
    return args_list

@contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    linger: int = 0,
    identity: Optional[bytes] = None,
    is_ipv6: bool = False,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket supporting IPv6
    Adapted from: vllm/vllm/utils.py
    """

    ctx = zmq.Context()  # type: ignore[attr-defined]
    if is_ipv6:
        ctx.setsockopt(zmq.IPV6,1)
    try:
        yield make_zmq_socket(ctx,
                              path,
                              socket_type,
                              bind=bind,
                              identity=identity)
    except KeyboardInterrupt:
        print("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)

def timeout(_func=None, *, seconds=5, throw_timeout_error=True):
    """
    Decorator to run a function with a timeout.

    Parameters:
        seconds (int or float): Timeout duration in seconds.
        throw_timeout_error : Throw timeout error if true
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            # In case the main program interpretor is dead, no new threads can be spawned.
            # Fallback to calling the func directly without timeout.
            if not threading.main_thread().is_alive():
                return func(*args, **kwargs)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError as e:
                    if throw_timeout_error:
                        raise TimeoutError(f"{func.__name__} timed out after {seconds} seconds.") from None
                    else:
                        print(f"{func.__name__} timed out after {seconds} seconds.")
                except Exception as e:
                    raise e
            return None
        return wrapper

    if callable(_func):
        return decorator(_func)

    return decorator

@timeout(seconds=30, throw_timeout_error=False)
def is_device_ready(qid):
    """
    Checks if device is ready and has atleast one NSP free

    Parameters:
        qid: Device QID.
    """
    from qaicrt import Util as qaic_util
    from qaicrt import QStatus, QDevStatus

    api_status, device_info = qaic_util().getDeviceInfo(qid)
    return (
        api_status == QStatus.QS_SUCCESS
        and device_info.devStatus == QDevStatus.QDS_READY
        and device_info.devData.resourceInfo.nspFree > 0
    )

def check_device_qid_avail(name: str, device_group: List[List[int]], _qids_inuse: set()):
    """Check device's availabilty for encode, prefill, or decode device group"""
    _num_qid = 0
    for qidList in device_group:
        _num_qid += len(qidList)
        for qid in qidList:
            _qids_inuse.add(qid)
            if not is_device_ready(qid):
                raise ValueError(f"Device id {qid} not available")
        if len(qidList) != len(device_group[0]):
            raise TypeError(f"Number of QIDs in {device_group} each {name} device group set should be same")
    return _num_qid

def run_args_mdp_generation_script(args: argparse.Namespace, output_json_path) -> None:
    """
    Constructs the command to run the MDP_Generator_Partitioner.py script
    based on the provided argparse.Namespace.
    """
    num_devices = str(len(args.device_group))
    num_partitions = str(args.override_qaic_config['stages'])

    # Base command: python3 MDP_Generator_Partitioner.py <required_positional_args>
    cmd_list = [
        sys.executable,  # Use the current Python executable
        os.path.join(importlib.util.find_spec("qaic_disagg").submodule_search_locations[0], 'MDP_Generator_Partitioner.py'),
        args.model, # model_name
        num_devices, # num_devices
        num_partitions, # num_partitions based on number of prefill stages.
        output_json_path, # output_json_path for the MDP generated
        "--prefill_seq_len", str(args.max_seq_len_to_capture),
        "--ctx_len", str(args.max_model_len),
        "--prefill_max_num_seqs", str(args.max_num_seqs),
    ]
    if args.quantization == "mxfp6":
        cmd_list.append("--mxfp6_matmul")
    if args.kv_cache_dtype == "mxint8":
        cmd_list.append("--mxint8_kv_cache")
    return cmd_list

def _run_partitioner(
    args: argparse.Namespace,
    tmp_mdp_json_path: str,
    partitioned_mdp_json_path: str,
) -> None:
    """
    Runs the Pipeline_partitioner.py script to partition a dumped MDP JSON.
    """
    num_devices = str(len(args.device_group))
    num_partitions = str(args.override_qaic_config['stages'])

    hf_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    layers_per_partition = hf_config.num_hidden_layers // int(num_partitions)

    # Step 2: Partition the dumped JSON using Pipeline_partitioner.py
    partition_cmd = [
        "python3",
        os.path.join(importlib.util.find_spec("qaic_disagg").submodule_search_locations[0], "Pipeline_partitioner.py"),
        tmp_mdp_json_path,
        partitioned_mdp_json_path,
        str(num_devices),
        str(num_partitions),
        str(layers_per_partition),
    ]
    print("Running Pipeline_partitioner.py...")
    try:
        subprocess.run(partition_cmd, check=True, capture_output=True, text=True)
        print(f"Partitioned JSON written to {partitioned_mdp_json_path}")
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise RuntimeError(f"Unexpected error while running Pipeline_partitioner {e.stderr}")

def reshape_and_encode_base64_image_from_url(content_url: str, height: Optional[int]=None, width: Optional[int]=None) -> str:
    """Encode a reshaped image retrieved from a remote url to base64 format."""
    image = Image.open(requests.get(content_url, stream=True).raw)

    # Resize the image to fixed dimension
    if height and width:
        resized_image = image.resize((width, height))
    else:
        resized_image = image

    # Save the resized image to a BytesIO buffer
    buffered = BytesIO()
    resized_image.save(buffered, format="JPEG")

    # Encode the image in base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_base64

def post_dummy_request(url: str, request_type: str, model_name: str, height: Optional[int]=None, width: Optional[int]=None):
    """Post dummy request by serving type at server launch."""
    if request_type == "text":
        response = requests.post(url,
                headers={"Content-Type": "application/json"},
                json={"model": model_name,
                    "prompt": "hi",
                    "max_tokens": 1,
                    "temperature": 0})

    elif request_type == "image+text":
        # sample multimodel client chat request
        text_prompt = "What are the animals in these images?"
        image_urls = ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"]

        image_base64_list = [
            reshape_and_encode_base64_image_from_url(url, height, width) for url in image_urls
        ]
        image_contents = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
            for image_base64 in image_base64_list
        ]
        prompt = dict(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text_prompt}, *image_contents],
                },
            ],
            model=model_name,
            max_completion_tokens=10,
        )
        response = requests.post(url, json=prompt)

    return response
