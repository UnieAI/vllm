# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Not a contribution.
# ---------------------------------------------------------------------------------------
#Adapted from vllm/examples/online_serving/disagg_examples/disagg_proxy_demo.py
"""
This file provides a disaggregated prefilling proxy XpYd disaggregated prefilling.
  python3 server.py  \
       --model $model_name  \
       --prefill localhost:8100 localhost:8101   \
       --decode localhost:8200 localhost:8201   \
       --port 8000
"""
# NOTE: Make sure all vllm imports are lazy-loaded to prevent triggering worker startup timeout in uvicorn
# Ref: https://github.com/Kludex/uvicorn/issues/2506

import argparse
import asyncio
import itertools
import json
import logging
import os
import ssl
import sys
from abc import ABC, abstractmethod
import base64
import torch
from io import BytesIO
from typing import Any, Callable, Optional, Dict
import threading
import aiohttp
import requests
import time

from vllm.utils.serial_utils import (
    MetadataItem,
    decode_pooling_output,
)
from fastapi import (APIRouter, Depends, Header, HTTPException,
                     Request, status)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from typing import List, Optional
from collections import defaultdict
from qaic_disagg.proxy.utils import validate_and_get_host, extract_host_port
from qaic_disagg.proxy.utils import (
    encode_tensor_to_base64,
    extract_mm_items,
    get_image_grid_thw,
    replace_mm_items
)

INSTANCE_REMOVE_THRESHOLD=5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def _get_headers_with_api_key() -> Dict:
    headers = {
            "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY')}"
        }
    return headers


class SchedulingPolicy(ABC):

    @abstractmethod
    def schedule(self, cycler: itertools.cycle, instances:Optional[List[str]] =None):
        pass
        raise NotImplementedError("Scheduling Proxy is not set.")

    @abstractmethod
    def post_schedule_update(self, instance:str):
        pass
        raise NotImplementedError("Post scheduling proxy is not set.")


class Proxy:
    def _is_model_qwen(self) -> bool:
        """Check if the model is a Qwen model"""
        return "Qwen" in self.model
    def __init__(
        self,
        encode_instances: list[str],
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        skip_disagg_prefill_threshold: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
    ):
        self.skip_prefill = (len(prefill_instances) == 0)
        self.encode_instances = encode_instances
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.encode_cycler = itertools.cycle(encode_instances)
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.skip_disagg_prefill_threshold = skip_disagg_prefill_threshold
        self.error_cnt_per_instance = defaultdict(lambda:0)
        self.tokenizer = None

        self.router = APIRouter()
        self.setup_routes()

        self.height = height
        self.width = width
        self.num_frames = num_frames
        if None not in (self.height, self.width) and self._is_model_qwen():
            self.image_grid_thw = get_image_grid_thw(
            self.model, self.height, self.width, self.num_frames
        )
        else:
            self.image_grid_thw = None

        from vllm.usage.usage_lib import is_usage_stats_enabled
        self.use_aggregated_usage_stats = is_usage_stats_enabled() and os.environ.get(
            "VLLM_DISAGG_USE_AGGREGATED_USAGE_STATS", "false").lower() in [
            "true", "1"
        ]

    def setup_routes(self):
        self.router.post(
            "/v1/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.create_completion)
        self.router.post(
            "/v1/chat/completions",
            dependencies=[Depends(self.validate_json_request)]
        )(self.create_chat_completion_epd)
        self.router.get("/status",
                        response_class=JSONResponse)(self.get_status)
        self.router.post("/instances/add",
                         dependencies=[Depends(self.api_key_authenticate)
                                       ])(self.add_instance_endpoint)

    # def dissagg_prefill_needed(self, request: Request):
    #     if self.tokenizer is None:
    #         return False

    #     return False

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail=
                "Unsupported Media Type: Only 'application/json' is allowed",
            )

    def api_key_authenticate(self, x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt with API Key: %s",
                           x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        # This requests done internally by the system to validate the instance
        # that is added. The new instance should share the same key set during
        # the start of disaggregated serving.
        headers = _get_headers_with_api_key()
        try:
            async with aiohttp.ClientSession(
                    timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.",
                                            instance)
                                return True
                            else:
                                logger.warning("Mismatch model %s : %s != %s",
                                               instance, model_cur, self.model)
                                return False
                        else:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(str(e))
            return False

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if not self.skip_prefill and (instance_type not in ["prefill", "decode"]) or \
                self.skip_prefill and (instance_type not in ["encode", "decode"]):
                raise HTTPException(status_code=400,
                                    detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance format.")
            try:
                host, port_str = extract_host_port(instance)
                host = validate_and_get_host(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400,
                                        detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance address.") from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(status_code=400,
                                    detail="Instance validation failed.")

            if instance_type == "encode":
                if instance not in self.encode_instances:
                    self.encode_instances.append(instance)
                    self.encode_cycler = itertools.cycle(
                        self.encode_instances)
                else:
                    raise HTTPException(status_code=400,
                                        detail="Instance already exists.")
            elif instance_type == "prefill":
                if instance not in self.prefill_instances:
                    self.prefill_instances.append(instance)
                    self.prefill_cycler = itertools.cycle(
                        self.prefill_instances)
                else:
                    raise HTTPException(status_code=400,
                                        detail="Instance already exists.")
            else:
                if instance not in self.decode_instances:
                    self.decode_instances.append(instance)
                    self.decode_cycler = itertools.cycle(self.decode_instances)
                else:
                    raise HTTPException(status_code=400,
                                        detail="Instance already exists.")

            return JSONResponse(content={
                "message":
                f"Added {instance} to {instance_type}_instances."
            })
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, headers={}, usage_stats=[]):
        use_chunked = data.get("stream", False)
        excluded_header_keys = ['content-length', 'host']
        headers = {k:v for k,v in headers.items() if k not in excluded_header_keys}
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)

        try:
            if use_chunked:
                generator = self.forward_request_streaming(url, data, headers, connector)
                return StreamingResponse(content=generator)
            else:
                response = await self.forward_request_non_streaming(url, data, headers, usage_stats, connector)
                return response
        except aiohttp.ClientError as e:
            logger.error("ClientError occurred: %s", str(e))
            await connector.close()
            raise HTTPException(
                status_code=502,
                detail=
                "Bad Gateway: Error communicating with upstream server.",
            ) from e
        except HTTPException as http_exc:
            logger.error("Unexpected error: %s", str(http_exc))
            await connector.close()
            raise http_exc
        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            await connector.close()
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request_streaming(self, url, data, headers, connector):
        async with aiohttp.ClientSession(connector=connector, timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=url, json=data,
                                    headers=headers) as response:
                if 200 <= response.status < 300 or 400 <= response.status < 500:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    error_content = await response.text()
                    try:
                        error_content = json.loads(error_content)
                    except json.JSONDecodeError:
                        error_content = error_content
                    logger.error("Request failed with status %s: %s",
                                    response.status, error_content)
                    raise HTTPException(
                        status_code=response.status,
                        detail=
                        f"Request failed with status {response.status}: "
                        f"{error_content}",
                    )

    async def forward_request_non_streaming(self, url, data, headers, usage_stats, connector):
        async with aiohttp.ClientSession(connector=connector, timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=url, json=data,
                                    headers=headers) as response:
                if 200 <= response.status < 300:
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/json" in content_type:
                        content = await response.json()
                        if self.use_aggregated_usage_stats and "usage" in content:
                            try:
                                if len(usage_stats) == 0:
                                    usage_stats.append(content["usage"])
                                else:
                                    prefill_usage = usage_stats[0]
                                    aggregated_usage = content["usage"]
                                    aggregated_usage["ttft_in_ms"] += prefill_usage["ttft_in_ms"]
                                    aggregated_usage["ttft_excluding_queue_wait_time_in_ms"] += prefill_usage["ttft_excluding_queue_wait_time_in_ms"]
                                    aggregated_usage["e2e_inference_in_ms"] += prefill_usage["ttft_in_ms"]
                                    aggregated_usage["queue_wait_time_in_ms"] += prefill_usage["queue_wait_time_in_ms"]
                                    aggregated_usage["e2e_inference_excluding_queue_wait_time_in_ms"] = aggregated_usage["e2e_inference_in_ms"] - aggregated_usage["queue_wait_time_in_ms"]
                            except Exception as e:
                                logger.error("Error occurred when aggregating usage stats: %s, returning the original usage stats", str(e))
                        return JSONResponse(content, status_code=response.status)
                    else:
                        raw = await response.read()
                        return Response(content=raw, status_code=response.status, media_type=content_type or "application/octet-stream")
                elif  400 <= response.status < 500:
                    # Request is not successful when the status code is between 400 and 500
                    # The proxy should forward vllm response
                    content_type = response.headers.get("content-type", "").lower()
                    error_content = await response.read()
                    error_string = error_content.decode("utf-8")
                    logger.error(
                        f"Request failed with status {response.status} {error_string} "
                    )
                    return Response(content=error_content, status_code=response.status, media_type=content_type or "application/octet-stream")
                else:
                    error_content = await response.text()
                    try:
                        error_content = json.loads(error_content)
                    except json.JSONDecodeError:
                        error_content = error_content
                    logger.error("Request failed with status %s: %s",
                                    response.status, error_content)
                    raise HTTPException(
                        status_code=response.status,
                        detail=
                        f"Request failed with status {response.status}: "
                        f"{error_content}",
                    )

    async def forward_encode(self, orig_request: dict, encode_instance: str):
        mm_items = extract_mm_items(orig_request)
        if not mm_items:
            return None # nothing to do

        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        async with aiohttp.ClientSession(connector=connector, timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

            target_url = encode_instance
            encoder_req = {
                "model": orig_request.get("model"),
                "messages": [
                    {"role": "user", "content": mm_items},
                ],
                "encoding_format": "bytes",
            }

            try:
                async with session.post(f"http://{target_url}/pooling", headers=headers,
                                        json=encoder_req) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:  # noqa: E501
                        metadata = json.loads(response.headers["metadata"])
                        body = await response.read()
                        items = [MetadataItem(**x) for x in metadata["data"]]

                        vision_embeds = decode_pooling_output(items=items, body=body)[0]
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error("Request failed with status %s: %s",
                                        response.status, error_content)
                        raise HTTPException(
                            status_code=response.status,
                            detail=
                            f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail=
                    "Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

            return vision_embeds, len(mm_items)

    def schedule(self, cycler: itertools.cycle, instances: Optional[List[str]] = None) -> str:
        return self.scheduling_policy.schedule(cycler, instances)

    def post_schedule_update(self, instance: str) -> str:
        return self.scheduling_policy.post_schedule_update(instance)

    async def get_status(self):
        if not self.skip_prefill:
            status = {
                "prefill_node_count": len(self.prefill_instances),
                "decode_node_count": len(self.decode_instances),
                "prefill_nodes": self.prefill_instances,
                "decode_nodes": self.decode_instances,
            }
        else:
            status = {
                "encode_node_count": len(self.encode_instances),
                "decode_node_count": len(self.decode_instances),
                "encode_nodes": self.encode_instances,
                "decode_nodes": self.decode_instances,
            }

        return status

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()
            # Forward the from header user's request to the vllm instance
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1
            kv_prepare_request["stream"] = False
            kv_prepare_request.pop("stream_options", None)

            skip_prefill = False
            # if self.skip_disagg_prefill_threshold:
            #     skip_prefill = self.dissagg_prefill_needed(kv_prepare_request)

            # prefill stage
            if not skip_prefill:
                prefill_instance = self.schedule(self.prefill_cycler, self.prefill_instances)
                try:
                    _ = await self.forward_request(
                            f"http://{prefill_instance}/v1/completions",
                            kv_prepare_request, headers=raw_request.headers)
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("prefill", prefill_instance)
                    raise http_exc

                self.post_schedule_update(prefill_instance)

            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler, self.decode_instances)

            try:
                response = await self.forward_request(
                    f"http://{decode_instance}/v1/completions", request, headers=raw_request.headers)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc

            self.post_schedule_update(decode_instance)

            return response
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg proxy server")
            print(error_messages)
            return StreamingResponse(content=iter(error_messages),
                            media_type="text/event-stream")

    async def create_chat_completion_epd(self, raw_request: Request):
        try:
            request = await raw_request.json()

            ## encode stage if applicable
            if len(self.encode_instances) > 0:
                encode_instance = self.schedule(self.encode_cycler, self.encode_instances)
                try:
                    encode_outputs = await self.forward_encode(request, encode_instance)
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("encode", encode_instance)
                    raise http_exc
                self.post_schedule_update(encode_instance)

                # process vision_embed to request format
                if isinstance(encode_outputs, tuple):
                    vision_embeds, processed_num_frames = encode_outputs
                    base64_image_embedding = encode_tensor_to_base64(vision_embeds)
                    if self._is_model_qwen():
                        embeds = {
                            "type": "image_embeds",
                            "image_embeds": {
                                "image_embeds": f"{base64_image_embedding}",  # Required
                            },
                            }
                        if self.image_grid_thw is not None:  # Required by Qwen
                            embeds["image_embeds"]["image_grid_thw"] = (
                                encode_tensor_to_base64(self.image_grid_thw[:processed_num_frames])
                            )
                    else:
                        embeds = {
                            "type": "image_embeds",
                            "image_embeds": f"{base64_image_embedding}",
                        }
                    # Update the original request with the new messages
                    updated_messages = replace_mm_items(request, embeds, processed_num_frames)
                    request["messages"] = updated_messages

            ## prefill stage
            usage_stats = []
            if not self.skip_prefill:
                kv_prepare_request = request.copy()
                kv_prepare_request["max_completion_tokens"] = 1
                kv_prepare_request["stream"] = False
                kv_prepare_request.pop("stream_options", None)

                prefill_instance = self.schedule(self.prefill_cycler, self.prefill_instances)
                try:
                    _ = await self.forward_request(
                            f"http://{prefill_instance}/v1/chat/completions",
                            kv_prepare_request,
                            usage_stats=usage_stats, headers=raw_request.headers)
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("prefill", prefill_instance)
                    raise http_exc

                self.post_schedule_update(prefill_instance)

            ## decode stage
            decode_instance = self.schedule(self.decode_cycler, self.decode_instances)

            try:
                response = await self.forward_request(
                    f"http://{decode_instance}/v1/chat/completions",
                    request, usage_stats=usage_stats, headers=raw_request.headers)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc

            self.post_schedule_update(decode_instance)

            return response

        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg proxy server")
            print(error_messages)
            return StreamingResponse(content=iter(error_messages),
                            media_type="text/event-stream")

    def remove_instance_endpoint(self, instance_type, instance):
        self.error_cnt_per_instance[instance] +=1
        logger.warning(f"Exception cnt for {instance_type} instance {instance} increased to {self.error_cnt_per_instance[instance]}")
        if self.error_cnt_per_instance[instance] >= INSTANCE_REMOVE_THRESHOLD:
            logger.warning(f"Due to higher number of exceptions {instance_type} instance {instance} removed from {instance_type} instances list")
            if (instance_type == "decode" and instance in self.decode_instances):
                self.decode_instances.remove(instance)
                self.decode_cycler = itertools.cycle(self.decode_instances)
            if (instance_type == "prefill" and instance in self.prefill_instances):
                self.prefill_instances.remove(instance)
                self.prefill_cycler = itertools.cycle(self.prefill_instances)
            if (instance_type == "encode" and instance in self.encode_instances):
                self.encode_instances.remove(instance)
                self.encode_cycler = itertools.cycle(self.encode_instances)

class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """
    Implements a round-robin scheduling policy for distributing requests across instances.
    """

    def __init__(self):
        super().__init__()

    def schedule(self, cycler: itertools.cycle, instances:Optional[List[str]] =None) -> str:
        return next(cycler)
    def post_schedule_update(self, instance):
        pass

class LeastOutstandingSchedulingPolicy(SchedulingPolicy):
    def __init__(self):
        super().__init__()
        self.instance_usage_outstanding = {}
        self.instance_usage_lock = threading.Lock()

    def schedule(self, cycler: itertools.cycle, instances:Optional[List[str]] =None) -> str:
        with self.instance_usage_lock:
            usage = {}
            for instance in instances:
                if instance not in self.instance_usage_outstanding:
                    self.instance_usage_outstanding[instance] = 0
                usage[instance] = self.instance_usage_outstanding[instance]
            min_key = min(usage, key=usage.get)
            # mark it as scheduled
            self.instance_usage_outstanding[min_key] += 1
            return min_key
    def post_schedule_update(self, instance):
        with self.instance_usage_lock:
            if instance in self.instance_usage_outstanding:
                self.instance_usage_outstanding[instance] -= 1

class ProxyServer:

    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: Optional[SchedulingPolicy] = None,
    ):
        """
        Initializes the ProxyServer with the provided arguments and optional custom functions.

        :param args: argparse.Namespace containing command-line arguments.
        :param scheduling_policy: An optional instance of SchedulingPolicy to determine how requests are distributed across instances.
        """
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.host = args.host
        self.workers = args.workers

        if scheduling_policy:
            self.scheduling_policy = scheduling_policy
        else:
            self.scheduling_policy = RoundRobinSchedulingPolicy() if args.router_policy == "round_robin" else LeastOutstandingSchedulingPolicy()

        self.proxy_instance = Proxy(
            encode_instances=[] if args.encode is None else args.encode,
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy= self.scheduling_policy,
            #skip_disagg_prefill_threshold= args.skip_disagg_prefill_threshold,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )

    def validate_parsed_serve_args(self, args: argparse.Namespace):
        # Decode is always required
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")

        # Either prefill or encode must be specified (but not necessarily both)
        if not args.prefill and not args.encode:
            raise ValueError("Please specify at least one prefill node or encode node.")

        # Validate instances only if they are provided
        for _type in (args.prefill, args.encode, args.decode):
            if _type:
                self.validate_instances(_type)

        # Verify model config only for provided instances
        if args.encode:
            self.verify_model_config(args.encode, args.model)
        if args.prefill:
            self.verify_model_config(args.prefill, args.model)
        self.verify_model_config(args.decode, args.model)

    def validate_instances(self, instances: list):
        for instance in instances:
            if not instance or ":" not in instance:
                raise ValueError(f"Invalid instance format: {instance}")
            try:
                host, port = extract_host_port(instance)
                host = validate_and_get_host(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(
                        f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(
                    f"Invalid instance {instance}: {str(e)}") from e

    def verify_model_config(self, instances: list, model: str) -> None:
        # Include api_key in request header if it is set during verification of model config
        headers = _get_headers_with_api_key()
        model_suffix = model.split("/")[-1]
        for instance in instances:
            try:
                response = requests.get(f"http://{instance}/v1/models", headers=headers)
                if response.status_code == 200:
                    model_cur = response.json()["data"][0]["id"]
                    model_cur_suffix = model_cur.split("/")[-1]
                    if model_cur_suffix != model_suffix:
                        raise ValueError(
                            f"{instance} serves a different model: "
                            f"{model_cur} != {model}")
                else:
                    raise ValueError(f"Cannot get model id from {instance}!")
            except requests.RequestException as e:
                raise ValueError(
                    f"Error communicating with {instance}: {str(e)}") from e

def parse_args():
    parser = argparse.ArgumentParser("QAIC vLLM disaggregated proxy server.")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument(
        "--encode",
        "-e",
        type=str,
        nargs="+",
        help="List of encode node URLs (host:port)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Image width",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Num of frames",
    )

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument("--host",
                        "-H",
                        type=str,
                        default="localhost",
                        help="Host address")

    parser.add_argument("--port",
                        "-P",
                        type=int,
                        default=8000,
                        help="Port to listen on")

    #Add argument to select scheduling policy between round robin or least outstanding
    parser.add_argument("--router-policy",
                        "-r",
                        type=str,
                        choices=["round_robin", "least_outstanding"],
                        default="round_robin",
                        help="Scheduling policy to use for requests")

    parser.add_argument("--workers",
                        "-w",
                        type=int,
                        default=None,
                        help="Number of uvicorn workers to use for the server")

    # parser.add_argument("--skip-disagg-prefill-threshold",
    #                     type=int,
    #                     default=None,
    #                     help="Skips disagg prefill if number of tokens is less than this threshold")

    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file.")

    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file.")

    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file.")

    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)."
    )

    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="Log level for uvicorn.")

    parser.add_argument("--disable-uvicorn-access-log",
                        action="store_true",
                        help="Disable uvicorn access log.")

    args = parser.parse_args()

    return args
