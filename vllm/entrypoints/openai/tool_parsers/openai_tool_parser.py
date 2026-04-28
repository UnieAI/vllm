# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Not a contribution.
# ---------------------------------------------------------------------------------------

# Temporarily adding PR #22386 [gpt-oss] tool parser supports for /chat/completions [1/n]
# for compatibility with function calling support from vLLM r0.10.2

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.harmony_utils import parse_output_into_messages
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer import AnyTokenizer


@ToolParserManager.register_module("openai")
class OpenAIToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support text-based extraction."  # noqa: E501
            )

        try:
            parser = parse_output_into_messages(token_ids)
        except Exception as e:
            raise ValueError(f"Failed to parse tokens from openai tool parser." )
        tool_calls = []
        final_content = None

        if len(parser.messages) > 0:
            for msg in parser.messages:
                if msg.recipient and msg.recipient.startswith("functions."):
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=msg.recipient.split("functions.")[1],
                                arguments=msg.content[0].text,
                            ),
                        ))
                elif msg.channel == "final":
                    final_content = msg.content[0].text

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=final_content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "Not being used, manual parsing in serving_chat.py"  # noqa: E501
        )