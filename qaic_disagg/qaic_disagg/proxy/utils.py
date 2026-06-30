# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------

# NOTE: Make sure all vllm imports are lazy-loaded to prevent triggering worker startup timeout in uvicorn
# Ref: https://github.com/Kludex/uvicorn/issues/2506

import base64
from io import BytesIO
import torch
from transformers import AutoConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from urllib.parse import urlparse

MM_TYPES = {"image_url", "audio_url", "input_audio"}

def validate_and_get_host(host, default_host: str = "0.0.0.0") -> str:
    """Get the host from the args object, or return the default host if not set."""
    import socket

    if host and host != "localhost":
        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror:
            raise TypeError("Invalid or unknown host")
        except Exception as e:
            raise TypeError(f"Error validating host. Error: {str(e)}")

    return host if host else default_host


def extract_host_port(url: str):
    """
    Extracts the host and port from a URL, including, hostnames, IPv4, IPv6, domain support.
    Args:
        url (str): The full URL (e.g., http://localhost:8000/endpoint)
    Returns:
        tuple: (host, port) or (None, None) if parsing fails
    """

    # Add dummy scheme if missing. urlparse works only for valid URLs
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        return host, port
    except Exception as e:
        raise ValueError(f"Error parsing endpoint {url} with error {e}")


def get_image_grid_thw(model_name: str, height: int, width: int, num_frames: int):
    height, width = smart_resize(height, width)
    config = AutoConfig.from_pretrained(model_name)
    patch_size = config.vision_config.patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]] * num_frames)
    return image_grid_thw


def encode_tensor_to_base64(tensor: torch.Tensor) -> str:
    buffer = BytesIO()
    torch.save(tensor.data, buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    return base64.b64encode(binary_data).decode("utf-8")


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.
    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


def replace_mm_items(request_data: dict, embeds: dict, num_frames: int) -> list[dict]:
    """
    Replace all image/audio items that appear anywhere in `messages` with the provided embeds.
    """
    updated_messages = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            updated_messages.append(msg) # TODO: check if sometimes, image exist solely (not in list)
            continue

        new_content = []
        embed_inserted = False
        for item in content:
            if item.get("type") in MM_TYPES:
                if not embed_inserted:
                    new_content.append(embeds)
                embed_inserted = True
            else:
                new_content.append(item)

        placeholder_inserted = False
        if embed_inserted:
            for item in new_content:
                # TODO: Image placeholders differ across models, so this needs to be updated to support multiple models.
                if item.get("type") == "text":
                    # Only one image_embed is allowed when sending a request to the OpenAI API.
                    # However, if the request contains multiple images,
                    # we want the updated prompt to contain multiple image placeholders.
                    # Therefore, we manually update the prompt based on the chat template.
                    item["text"] = "<|vision_start|><|image_pad|><|vision_end|>" * (num_frames - 1) + item["text"]
                    placeholder_inserted = True
                    break # Only modify the first text item

            # create text section if it's not already exist
            if not placeholder_inserted:
                new_content.append({"type": "text", "text":"<|vision_start|><|image_pad|><|vision_end|>" * (num_frames - 1)})
                placeholder_inserted = True

        msg["content"] = new_content
        updated_messages.append(msg)
    return updated_messages
