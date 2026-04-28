# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Utilities for selecting and loading qaic models."""
import importlib
import os, json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List, Any, Callable
import time
import signal
import torch
import torch.nn as nn
import numpy as np
import transformers
import requests
from multiprocessing import Queue
from transformers import PretrainedConfig, AutoModelForCausalLM
from vllm.config import VllmConfig, ModelConfig, PoolerConfig
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler, PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from peft import PeftConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.qserve_model_runner import QServeModelRunner
from vllm.logger import init_logger
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.utils import cdiv
logger = init_logger(__name__)

VLLM_CACHE_DTYPE_TO_QAIC_CACHE_DTYPE = {
    "auto": "fp16",
    "fp8": "mxint8",
    "mxint8": "mxint8"
}

single_qpc_config = {
    "qpc_path": os.environ.get("VLLM_QAIC_QPC_PATH", None),
    "mos": os.environ.get("VLLM_QAIC_MOS", None),
    "aic_enable_depth_first": os.environ.get("VLLM_QAIC_DFS_EN", None),
    "device_group": os.environ.get("VLLM_QAIC_QID", None),
    "num_cores": os.environ.get("VLLM_QAIC_NUM_CORES", None),
    "compiler_args": os.environ.get("VLLM_QAIC_COMPILER_ARGS", None)
}
QAIC_DEVICE_CONFIG = {
    "target": {
        "qpc_path": os.environ.get("VLLM_QAIC_SPEC_TARGET_QPC_PATH", None),
        "mos": os.environ.get("VLLM_QAIC_SPEC_TARGET_MOS", None),
        "aic_enable_depth_first": os.environ.get("VLLM_QAIC_SPEC_TARGET_DFS_EN", None),
        "device_group": os.environ.get("VLLM_QAIC_SPEC_TARGET_QID", None),
        "num_cores": os.environ.get("VLLM_QAIC_SPEC_TARGET_NUM_CORES", None),
        "compiler_args": os.environ.get("VLLM_QAIC_SPEC_TARGET_COMPILER_ARGS", None)
    },
    "draft":{
        "qpc_path": os.environ.get("VLLM_QAIC_SPEC_DRAFT_QPC_PATH", None),
        "mos": os.environ.get("VLLM_QAIC_SPEC_DRAFT_MOS",None),
        "aic_enable_depth_first": os.environ.get("VLLM_QAIC_SPEC_DRAFT_DFS_EN", None),
        "mxint8_kv_cache": os.environ.get("VLLM_QAIC_SPEC_DRAFT_KV_COMPRESSION", None),
        "device_group": os.environ.get("VLLM_QAIC_SPEC_DRAFT_QID", None),
        "num_cores": os.environ.get("VLLM_QAIC_SPEC_DRAFT_NUM_CORES", None),
        "compiler_args": os.environ.get("VLLM_QAIC_SPEC_DRAFT_COMPILER_ARGS", None)
    },
    "default": single_qpc_config,
    "turbo": single_qpc_config
}


@dataclass
class QaicCompileConfig:
    compile_only: bool
    qpc_path: str
    device_group: List[int]
    cfg: Dict[str, Any]
    num_logits_to_keep: Optional[int]
    kv_offload: bool
    qpc_idx: Optional[int]
    include_sampler: Optional[bool]
    include_guided_decoding: Optional[bool]
    return_pdfs: Optional[bool]
    max_top_k_ids: Optional[int]
    qaic_config: Optional[Dict[str, Any]] = None
    stages: Optional[int] = 1,
    queue: Optional[Queue] = None

class QaicCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig
    ) -> None:
        super().__init__()
        self.vocab_size = vllm_config.model_config.hf_config.get_text_config().vocab_size
        self.vllm_config = vllm_config
        self.sampler = Sampler()
        self.logits_processor = LogitsProcessor(self.vocab_size,
                                        logits_as_input=True)
        # Lazy initialized
        self.model: nn.Module
        pooler_config = vllm_config.model_config.pooler_config
        self._pooler = None
        if vllm_config.model_config.runner_type == "pooling" and not vllm_config.model_config.is_multimodal_model:
            assert pooler_config is not None
            self._pooler = DispatchPooler({
                "encode": Pooler.for_encode(pooler_config),
                "embed": Pooler.for_embed(pooler_config),
                "classify": Pooler.for_classify(pooler_config, None),
                "score": Pooler.for_classify(pooler_config, None),
            })
        # below variables are specific for turbo run

        # init variables to hold speculations for prefill/decode
        self.prefill_proposer_scores: Optional[torch.Tensor] = None # shape: [bs, num_speculative_tokens, vocab_size]
        self.decode_proposer_scores: Optional[torch.Tensor] = None # shape: [bs, precode_len, num_speculative_tokens, vocab_size]
        # init model `hidden_states` to encode number of accepted tokens after decode pass
        self.hidden_states: Optional[torch.Tensor] = None # shape: [bs, precode_len, vocab_size]
        self.accepted_hidden_states: Optional[torch.Tensor] = None # shape: [bs, vocab_size]
        # init tracking of sequence ids to prune sequences that were moved to finish during StopChecker
        # this is needed for our implementation because vllm does this pruning before `MQAScorer` with `hidden_states`
        # since our speculations in `self.prefill_proposer_scores` and `self.decode_propoers_scores` do not receive this pruning,
        # we have to do it to align
        self.seq_ids: List[int] = []
        self.current_decode_seq_ids = []
        if vllm_config.speculative_config is not None and vllm_config.speculative_config.method == "turbo":
            num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
            bs = vllm_config.scheduler_config.max_num_seqs
            precode_len = num_speculative_tokens+1
            self.hidden_states = torch.arange(precode_len).view(1, precode_len, 1).expand(bs, precode_len, 1)
            self.extracted_hidden_states = self.hidden_states[:,0]
            self.batch_indices = torch.arange(bs)

    def forward(
        self,
        input_ids: List[np.ndarray],
        positions: List[np.ndarray],
        batch_indices: List[int],
        is_prompt: bool,
        lora_ids: Optional[List[np.ndarray]] = None,
        sampling_params: Optional[Dict[str, Union[List[float], List[int], List[np.ndarray]]]] = None,
        bypass_model_exec: Optional[bool] = False,
        kv_caches: Optional[List[List[np.ndarray]]] = None,
        logits_mem_buffs: Optional[List[np.ndarray]] = None,
        callback: Optional[Callable] = None,
        multi_modal_kwargs_list: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        if is_prompt:
            qserve_inputs = {}
            logits_list = []
            for index, (iids,pids,bids) in enumerate(zip(input_ids, positions, batch_indices)):
                inputs = {
                    'input_ids':iids,
                    'position_ids':pids,
                    'batch_index':np.array([[bids]])
                }
                if kv_caches:
                    inputs['kv_cache'] = kv_caches[index]
                if logits_mem_buffs:
                    inputs['logits'] = logits_mem_buffs[index]   
                if lora_ids:
                    inputs['lora_ids'] = np.array([[lora_ids[index]]])

                if bypass_model_exec:
                    inputs['ignore_prefill'] = bypass_model_exec

                if sampling_params:
                    for k, v in sampling_params.items():
                        if isinstance(v[index], np.ndarray):
                            inputs[k] = v[index]
                        elif v[index] is not None:
                            inputs[k] = np.asarray([[v[index]]],
                                dtype=np.int32 if isinstance(v[index], int) else np.float32)

                if multi_modal_kwargs_list and multi_modal_kwargs_list[index] is not None:
                    for k, v in multi_modal_kwargs_list[index].items():
                        if v is not None:
                            inputs[k] = v
                qserve_inputs[bids] = inputs
            logits = self.model.run(qserve_inputs, is_prompt, callback)
            return torch.from_numpy(logits)

        qserve_inputs = {}
        for index, (iids,pids,bids) in enumerate(zip(input_ids, positions, batch_indices)):
            inputs = {
                'input_ids':iids,
                'position_ids':pids
            }
            if lora_ids:
                inputs['lora_ids'] = np.array([[lora_ids[index]]])
            if sampling_params:
                for k, v in sampling_params.items():
                    if isinstance(v[index], np.ndarray):
                        inputs[k] = v[index]
                    elif v[index] is not None:
                        inputs[k] = np.asarray([[v[index]]],
                            dtype=np.int32 if isinstance(v[index], int) else np.float32)
            qserve_inputs[bids] = inputs

        logits = self.model.run(qserve_inputs, is_prompt, callback) # non-write
        # logits is a non-writable array. pytorch needs to have a
        # writable array to work properyly (else, behavior is undefined)
        # https://python-code.dev/articles/413443632
        return torch.from_numpy(np.copy(logits))

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        if not self._pooler:
            raise ValueError("Something went wrong, pooler is not supposed to be called")
        return self._pooler(hidden_states, pooling_metadata)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def process_logits(self, logits: torch.Tensor,
                       speculative_model_type: str,
                       is_prompt: bool,
                       sampling_metadata: SamplingMetadata,
                       ) -> torch.Tensor:
        if len(logits.shape) > 2:
            if speculative_model_type == "target":
                # reshape the output logits for TLM to simulate batch-expansion in case of precode
                logits = logits.reshape((logits.shape[0]*logits.shape[1],-1))
            elif speculative_model_type == "turbo":
                # separate base model outputs (`logits`) from speculations (`self.proposer_scores`)
                # later, once `self.proposer_worker.get_spec_proposals(.)` gets called during `SpecDecodeWorker:_run_speculative_decoding_step(.)``
                # the saved speculations will be used to generate proposals
                if is_prompt:
                    # prefill implies that new sequences are being added so we must track seq ids
                    seq_ids = [sg.seq_ids[0] for sg in sampling_metadata.seq_groups]
                    self.seq_ids.extend(seq_ids)
                    # cache speculation predictions to be used during `generate_proposals(.)` phase
                    prefill_proposer_scores = logits[:, 1:] # shape: [bs, num_speculative_tokens, vocab_size] if prefill
                    if self.prefill_proposer_scores is None:
                        # happens on 1st step of llm_engine or when a prefill request is scheduled when previous was a decode run
                        self.prefill_proposer_scores = prefill_proposer_scores
                    else:
                        # happens when two or more prefills are scheduled back-to-back (meaning no decode iterations in between)
                        # e.g., prefill_scheduled -> incoming_requests_is_received -> prefill_scheduled
                        self.prefill_proposer_scores = torch.cat([self.prefill_proposer_scores, prefill_proposer_scores]) # shape: [bs+n, num_speculative_tokens, vocab_size]
                    # extract base model predictions
                    logits = logits[:, 0] # shape: [batch_size, vocab_size]
                else:
                    assert self.prefill_proposer_scores is None
                    # cache decode phase speculative proposals to be used during `generate_proposals(.)` phase
                    self.decode_proposer_scores = logits[:, :, 1:] # shape: [bs, num_speculative_tokens+1, num_speculative_tokens, vocab_size]
                    # extract base model predictions
                    logits = logits[:, :, 0] # shape: [batch_size, num_speculative_tokens+1, vocab_size]
                    # batch expansion of logits
                    bs, precode_len = logits.shape[:2]
                    logits = logits.view(bs*precode_len, -1)
            else:
                raise ValueError(f"we do not exepct to get logits with rank {logits.ndim} on non-spd scenario.")
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def process_sample(self, sample: Optional[SamplerOutput], is_prompt: bool) -> Optional[SamplerOutput]:
        if self.prefill_proposer_scores is not None or self.decode_proposer_scores is not None:
            # should only be true if running with turbo
            bs = len(sample.outputs)
            if is_prompt:
                # insert dummy hidden_states so that SpecDecodeWorker has a `self.previous_hidden_states` to extract
                # or to register seqid into existing `self.previous_hidden_states` of `SpecDecodeWorker`
                sample.hidden_states = self.extracted_hidden_states[:bs] # shape: [bs, vocab_size]
            else:
                # should be True during `MQAScorer`, the `self.hidden_states` encodes number of accepted tokens
                # to be used during `generate_proposals` to extract the correct sequence idx speculations
                sample.hidden_states = self.hidden_states[:bs] # shape: [bs, num_speculative_tokens+1, vocab_size]

        return sample

    def load_model(self, **kwargs):
        # Transform model to be QEfficient compliant
        # from QEfficient.cloud.export import main
        # main(model_name, cache_dir)
        model = QServeModelRunner(**kwargs)
        self.model = model

    def kv_cache_info(self):
        no_buff_l = []
        for name in self.model.session.input_names:
            if name.startswith("past_key") or name.startswith("past_value"):
                no_buff_l.append(name)
        return [self.model.get_input_shape_and_dtype(no_buff_l[0]), len(no_buff_l)]

    def get_allowed_seqlens(self):
        allowed_seqlens = []
        for allowed_shape in self.model.session.allowed_shapes:
            allowed_seqlens.append(allowed_shape[1][1][1])
        return allowed_seqlens

    def generate_proposals(self, previous_hidden_states, sampling_metadata: SamplingMetadata) -> List[SamplerOutput]:
        # currently, only turbo/medusa use this method
        # filter out finished batches similar to `HiddenStates.prune(.)` does it
        seq_ids = [sg.seq_ids[0] for sg in sampling_metadata.seq_groups]
        if self.prefill_proposer_scores is not None and self.decode_proposer_scores is None:
            # should only be true when scheduling first decode iteration
            ndim = self.prefill_proposer_scores.ndim
            assert ndim == 3
            chosen_proposer_scores = self.prefill_proposer_scores
        else:
            # should be true for all decode loops after the first one
            assert self.decode_proposer_scores is not None
            # extract precode len and vocab size
            precode_len = self.decode_proposer_scores.shape[0]
            # check if any seqs from `self.seq_ids` were pruned (moved from `running` -> `finish`)
            if seq_ids != self.seq_ids:
                # extract all decode sequences that are in `running` state
                unpruned_decode_bs = self.decode_proposer_scores.shape[0]
                index = []
                for s_id in seq_ids:
                    if s_id in self.seq_ids:
                        # s_id is in `running` state
                        idx = self.seq_ids.index(s_id)
                        # ignore decode seq if it was previously prefill (they will be filtered out from `previous_hidden_states` later)
                        if idx <= unpruned_decode_bs-1:
                            index.append(idx)
                self.decode_proposer_scores = self.decode_proposer_scores[index]
                self.seq_ids = seq_ids
                decode_bs = len(index)
            else:
                decode_bs = self.decode_proposer_scores.shape[0]
            # extract decode previous hidden states (ignore prefill hidden states which are located at rightmost of batch index if they exist)
            decode_previous_hidden_states = previous_hidden_states[:decode_bs]
            # previous_hidden_states.shape: [bs, vocab_size]
            previous_hidden_states_ndim = previous_hidden_states.ndim
            assert previous_hidden_states_ndim == 2
            # extract number of accepted tokens idx encoded by den_states` `
            chosen_proposer_scores = self.decode_proposer_scores[self.batch_indices[:decode_bs], decode_previous_hidden_states.squeeze(1)] # shape: [bs, num_speculative_tokens, vocab_size]
            # if previous iteration was a prefill, concatenate decode proposals with prefill proposals
            if self.prefill_proposer_scores is not None:
                chosen_proposer_scores = torch.cat([chosen_proposer_scores, self.prefill_proposer_scores]) # shape: [bs', num_speculative_tokens, vocab_size]
        # reset prefill propossals
        self.prefill_proposer_scores = None
        # sample speculations
        sampled_token_ids = chosen_proposer_scores.argmax(-1) # shape: [bs, num_speculative_tokens]
        sampled_token_probs = torch.softmax(chosen_proposer_scores, -1) # shape: [bs, num_speculative_tokens, vocab_size]
        logprobs = torch.log_softmax(chosen_proposer_scores, -1) # shape: [bs, num_speculative_tokens, vocab_size]
        # wrap speculations in SamplerOutput
        outputs: List[SamplerOutput] = []
        for batch_idx in range(len(sampling_metadata.seq_groups)):
            output = SamplerOutput(
                outputs = None,
                sampled_token_probs=sampled_token_probs[batch_idx],
                logprobs=logprobs[batch_idx],
                sampled_token_ids=sampled_token_ids[batch_idx]
            )
            outputs.append(output)
        return outputs

def _check_model_architecture(config: PretrainedConfig) -> None:
    from QEfficient.transformers.modeling_utils import get_lists_of_cb_qeff_models
    architectures = getattr(config, "architectures", [])

    for arch in architectures:
        if arch in get_lists_of_cb_qeff_models.architectures:
            return
    raise ValueError(
        f"Model architectures {architectures} are not supported in QEfficient transformer for qaic "
        f"for now. Supported architectures: "
        f"{list(get_lists_of_cb_qeff_models.architectures)}")


def check_qpc_exists(qpc_path:str) -> bool:

    bin_path = os.path.join(qpc_path, "programqpc.bin")
    if not os.path.exists(qpc_path):
        return False
    if not os.path.isdir(qpc_path):
        if "programqpc.bin" in qpc_path:
            return True

    return os.path.exists(bin_path)

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False

def get_hf_model(model_config: ModelConfig, qaic_config: Optional[dict] = None, lora_config=None, kv_offload=False):
    logger.info("Downloading model from Hugging face server")

    from QEfficient import QEFFAutoModelForCausalLM
    from QEfficient.peft.lora import QEffAutoLoraModelForCausalLM
    from QEfficient import QEFFAutoModelForImageTextToText
    from QEfficient import QEFFAutoModelForSpeechSeq2Seq
    from QEfficient import QEFFAutoModel
    QEff_class_mapping = {
        'default': QEFFAutoModelForCausalLM,
        'lora': QEffAutoLoraModelForCausalLM,
        'imagetext': QEFFAutoModelForImageTextToText,
        'speech': QEFFAutoModelForSpeechSeq2Seq,
        'encode': QEFFAutoModel,
    }

    hf_config = model_config.hf_config
    if hf_config.model_type in _CONFIG_REGISTRY or not is_json_serializable(hf_config):
        # If vllm uses a custom model config class,
        # convert it back to the transformers config class
        from transformers import AutoConfig
        pretrained_hf_config = AutoConfig.from_pretrained(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
        )
        hf_config = AutoConfig.from_pretrained(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            **hf_config.to_dict(),
        )
        # If tie_word_embeddings is not set correctly,
        # single QPC's output would be wrong
        hf_config.tie_word_embeddings = pretrained_hf_config.tie_word_embeddings
    args = {
        "continuous_batching": model_config.override_qaic_config.get('continuous_batching', True),
        "qaic_config": qaic_config,
        "trust_remote_code": model_config.trust_remote_code,
        "revision": model_config.revision,
        "code_revision": model_config.code_revision,
        "attn_implementation": "eager",
        "config": hf_config,
        "kv_offload": kv_offload,
    }

    model_type = 'lora' if lora_config else 'default'

    if model_config.is_multimodal_model:
        if speech_cfg := getattr(model_config.hf_config, "max_source_positions", None):
            model_type = "speech"
            del args["kv_offload"]
        elif model_config.is_multimodal_model and model_config.hf_config.model_type != "internvl_chat":
            model_type = 'imagetext'
        # Speculative decoding not supported with multimodality
        # On-device sampling is supported with multimodality
        # Continuous batching is supported for dual QPC VLMs except for llama3.2
        # Continuous batching is not supported for single QPC approach and audio models
        if not kv_offload or "Llama-3.2" in model_config.model or model_type == "speech":
            del args["continuous_batching"]
        if model_type == "speech":
            del args["qaic_config"]

    if model_config.runner_type == "pooling" and not model_config.is_multimodal_model:
        model_type = 'encode'
        del args["continuous_batching"]
        del args["qaic_config"]
        del args["kv_offload"]
        if model_config.override_qaic_config:
            if 'pooling_device' in model_config.override_qaic_config:
                if model_config.override_qaic_config['pooling_device'] == 'qaic':
                    args["pooling"] = model_config.override_qaic_config['pooling_method']

    max_retries = 8
    retry_count = 0
    import requests
    while retry_count < max_retries:
        try:
            model_hf = QEff_class_mapping[model_type].from_pretrained(
                model_config.model,
                **args
            )
            break
        except requests.ReadTimeout as e:
            logger.info(f"HF hub read timeout: {e}")
            retry_count += 1
        except requests.exceptions.HTTPError as e:
            retry_count = max_retries
            if e.response.status_code == 401:
                logger.error("You need to set HF_TOKEN environment"
                            " variable to download private"
                            " checkpoints.")
            else:
                raise e
        except Exception as e:
            logger.warning(f"Unable to access HF hub due to an exception: {e}")
            retry_count += 1

    if retry_count >= max_retries:
        raise ValueError(
        f"Unable to download model {model_config.model} from Hugging face!")

    return model_hf

def _get_qaic_compile_config(
    vllm_config: VllmConfig,
    speculative_model_type: str = "default",
) -> QaicCompileConfig:

    def _clean_config(cfg:Dict[str, Any])->Dict[str, Any]:
        update_cfg = {}

        if cfg is None:
            return {}

        # compiler args
        if "compiler_args" in cfg:
            import re
            if cfg["compiler_args"] is not None:
                vl = re.split(' |\|', cfg["compiler_args"])
                for v in vl:
                    v = v.split("=")
                    if len(v) == 1 :
                        cfg[v[0]] = True
                    else:
                        cfg[v[0]] = v[1]
            del cfg["compiler_args"]
        # Fix key names
        _cfg = {}
        for key in cfg.keys():
            key1 = key.lower().replace('-','_').strip()
            _cfg[key1] = cfg[key]
        cfg = _cfg

        # Clean override config
        for key in cfg.keys():
            value = cfg[key]
            if value is not None:
                #key = key.lower().replace('-','_')
                if isinstance(value, (str,bool)):
                    if key != "qpc_path" and key != "mdp_load_partition_config" and key != "aic_pmu_recipe" and key != "mdp_dump_partition_config":
                        value = str(value).lower()
                    value = str(value).strip()

                # Ignore donot update list
                _ignore_list = [ "prefill_seq_len", "ctx_len","batch_size",
                                            "full_batch_size", "num_speculative_tokens"]
                if key in _ignore_list:
                    continue
                # specific filters
                # num_device
                if key in ["device_id", "device_group", "device_ids"]:
                    if isinstance(value, str):
                        #value = value.replace('[','').replace(']','')
                        #value = value.split(',')
                        import re
                        value  =  re.sub(r'[^0-9]', ' ', value).strip()
                        value  =  re.sub(r' +', ',', value).split(',')
                    if isinstance(value, int):
                        value = [value]
                    value = [int(v) for v in value]
                    update_cfg["device_group"]  = value
                    update_cfg["num_devices"] = len(value)
                # num_cores
                elif key in ["num_cores","aic_num_cores"]:
                    update_cfg["num_cores"] = int(value)
                # num_devices
                elif key in ["num_devices"]:
                    update_cfg["num_devices"] = int(value)
                #mxfp6
                elif (key in ["mxfp6", "mxfp6_matmul", "mxfp6_en"] or
                    (value == "mxfp6")):
                    update_cfg["mxfp6_matmul"] = True if value not in ["false", "0"] else False
                #mxint8
                elif (key in ["mxint8", "mxint8_en", "mxint8_kv_cache"] or
                    (value == "mxint8")):
                    update_cfg["mxint8_kv_cache"] = True if value not in ["false", "0"] else False
                #dfs
                elif key in ["dfs", "aic_enable_depth_first"]:
                    update_cfg["aic_enable_depth_first"] = True if value not in ["false", "0"] else False
                # mos
                elif key == "mos":
                    update_cfg["mos"] = int(value)
                elif key == "mdts_mos":
                    update_cfg[key] = int(value)
                #Anything else will pass as it is
                elif value in ["", "true", "1"]:
                    update_cfg[key] = True
                elif value in ["false", "0"]:
                    update_cfg[key] = False
                elif key in ["num_patches", "height", "width"]:
                    update_cfg[key] = int(value)
                elif key == "embed_seq_len":
                    if isinstance(value,str):
                        value = value.strip().split(',')
                        value = list(map(int, value))
                    elif isinstance(value, int):
                        assert value == vllm_config.model_config.max_model_len, "sequence length should be the same as max_model_len"
                    assert vllm_config.model_config.max_model_len in value, "max_model_len should be passed in embed_seq_len"
                    update_cfg["prefill_seq_len"] = value
                elif key in ["comp_ctx_lengths_prefill", "comp_ctx_lengths_decode"]:
                    try:
                        if isinstance(value, str):
                            value = value.strip().split(',')
                        value = list(map(int, value))
                        assert len(value)>0, f"{key} should be non-empty"
                        assert all(v<=vllm_config.model_config.max_model_len  for v in value), "All values of comp_ctx_lengths must be integers and less than max_model_len"
                        value.sort()
                        update_cfg[key] = value
                    except:
                        logger.warning("Compute Context Lengths not found")
                else: # For other compiler args
                    update_cfg[key] = value

        return update_cfg

    mxfp6_en, mxint8_en = False, False

    #mxfp6
    if isinstance(vllm_config.model_config.quantization,str) and\
            vllm_config.model_config.quantization == "mxfp6":
            mxfp6_en = True
    #mxint8
    if "mxint8" in\
            VLLM_CACHE_DTYPE_TO_QAIC_CACHE_DTYPE[vllm_config.cache_config.cache_dtype]:
        mxint8_en = True
    # Number of kv cache blocks should be same as num_gpu_blocks, if CPL==blk_size
    kv_cache_batch_size = vllm_config.cache_config.num_cpu_blocks

    prefill_only = None
    # prefill_only options
    if vllm_config.kv_transfer_config:
        kv_role = vllm_config.kv_transfer_config.kv_role
        if kv_role in ["kv_producer", "kv_consumer"]:
            prefill_only = kv_role == "kv_producer"

    if vllm_config.model_config.runner_type == "pooling" and not vllm_config.model_config.is_multimodal_model:
        vllm_config.model_config.max_seq_len_to_capture = vllm_config.model_config.max_model_len

    # Prepare default config
    cfg = {
        "qpc_path": None,
        "prefill_seq_len": vllm_config.model_config.max_seq_len_to_capture,
        "ctx_len": vllm_config.scheduler_config.max_model_len,
        "batch_size": 1,
        "full_batch_size": vllm_config.scheduler_config.max_num_seqs,
        "kv_cache_batch_size": kv_cache_batch_size,
        "device_group": vllm_config.device_config.device_group,
        "num_devices": len(vllm_config.device_config.device_group) if vllm_config.device_config.device_group is not None else 1,
        "num_cores": None,
        "mxfp6_matmul": mxfp6_en,
        "mxint8_kv_cache": mxint8_en,
        "num_speculative_tokens": None,
        "aic_enable_depth_first": True,
        "mos": -1,
        "prefill_only": prefill_only,
        "compile_only": False,
    }

    # update default settings
    cfg.update(_clean_config(vllm_config.model_config.override_qaic_config))
    vllm_config.model_config.override_qaic_config = _clean_config(vllm_config.model_config.override_qaic_config)
    # update through environment variable
    cfg.update(_clean_config(QAIC_DEVICE_CONFIG[speculative_model_type]))

    # set aic num core as per the hw if not provided
    if cfg["num_cores"] is None:
        _hw_num_cores = 16
        from qaicrt import Util as qaic_util
        from qaicrt import QStatus
        if cfg["device_group"] is not None:
            for id in cfg["device_group"]:
                _nsp_info = qaic_util().getResourceInfo(id)
                if _nsp_info[0] != QStatus.QS_SUCCESS:
                    raise ValueError(f"device_id {id} is not available !!")
                _hw_num_cores = min(_hw_num_cores, _nsp_info[1].nspTotal)

        cfg["num_cores"] = _hw_num_cores
        # Applicable for draft-target spd scheme
        if vllm_config.speculative_config and "draft" in vllm_config.speculative_config.method:
            other_cfg = {"device_group":cfg["device_group"]}
            if speculative_model_type =="target" and vllm_config.speculative_config.draft_override_qaic_config is not None:
                    other_cfg.update(_clean_config(vllm_config.speculative_config.draft_override_qaic_config))
                    other_cfg.update(_clean_config({"device_group":os.environ.get("VLLM_QAIC_SPEC_DRAFT_QID", None)}))
            else:
                # update default settings
                other_cfg.update(_clean_config(vllm_config.speculative_config.target_model_config.override_qaic_config))
                other_cfg.update(_clean_config({"device_group":os.environ.get("VLLM_QAIC_SPEC_TARGET_QID", None)}))
            if other_cfg["device_group"] == cfg["device_group"]:
                _targetCoreCount = cdiv(_hw_num_cores , 2)
                if speculative_model_type =="target":
                    cfg["num_cores"] = _targetCoreCount
                else:
                    cfg["num_cores"] = _hw_num_cores - _targetCoreCount

    if not vllm_config.cache_config.enable_prefix_caching:
            del cfg["kv_cache_batch_size"]

    if cfg["mos"] == -1:
        del cfg["mos"]

    num_logits_to_keep = None
    if speculative_model_type in ("target", "turbo"):
        cfg["num_speculative_tokens"] = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else None
        num_logits_to_keep =  vllm_config.speculative_config.num_speculative_tokens + 1
    else:
        del cfg["num_speculative_tokens"]

    if vllm_config.model_config.is_multimodal_model:
        support_continuous_batching = False
        if vis_cfg := getattr(vllm_config.model_config.hf_config, "vision_config", None):
            if cfg.get("kv_offload", False) and "Llama-3.2" not in vllm_config.model_config.model:
                # Single QPC approach does not support continuous batching
                # Llama 3.2 does not support continuous batching yet
                support_continuous_batching = True
            if "height" not in cfg:
                cfg["img_size"] = getattr(vis_cfg, "image_size", 448)
            if max_dynamic_patch := getattr(
                vllm_config.model_config.hf_config, "max_dynamic_patch", None
            ):
                if "num_patches" not in cfg:
                    use_thumbnail = vllm_config.model_config.hf_config.use_thumbnail
                    cfg["num_patches"] = max_dynamic_patch + int(use_thumbnail)
            if vllm_config.model_config.mm_processor_kwargs is not None:
                if "max_patches" in vllm_config.model_config.mm_processor_kwargs:
                    cfg["max_num_tiles"] = vllm_config.model_config.mm_processor_kwargs["max_patches"] + 1
            num_frames = vllm_config.model_config.multimodal_config.get_limit_per_prompt("image")
            if num_frames > 1:
                cfg["num_frames"] = num_frames
        else:
            # Audio models don't support continuous batching yet
            if "encoder_ctx_len" not in cfg:
                cfg["encoder_ctx_len"] = getattr(vllm_config.model_config.hf_config, "max_source_positions", None)

        if not support_continuous_batching:
            assert cfg["full_batch_size"] == 1, (
                "Multimodal models do not support batching yet. "
                "Please set `max_num_seqs` (decode batch size) to 1."
            )
            del cfg["full_batch_size"]

    qaic_config = None
    if speculative_model_type in ("target", "turbo"):
        qaic_config = dict(speculative_model_type=speculative_model_type)

    # On Device Sampling
    if cfg.get("aic_include_sampler", None) is not None:
        if qaic_config is None:
            qaic_config = dict()
        qaic_config["include_sampler"] = cfg["aic_include_sampler"]
        if cfg.get("aic_return_pdfs", None) is not None:
            qaic_config["return_pdfs"] = cfg["aic_return_pdfs"]
            del cfg["aic_return_pdfs"]
        if cfg.get("max_top_k_ids", None) is not None:
            qaic_config["max_top_k_ids"] = min(
                int(cfg["max_top_k_ids"]), vllm_config.model_config.get_vocab_size()
            )
            del cfg["max_top_k_ids"]
        if cfg.get("aic_include_guided_decoding", None) is not None:
            qaic_config["include_guided_decoding"] = cfg["aic_include_guided_decoding"]
            del cfg["aic_include_guided_decoding"]
        del cfg["aic_include_sampler"]

    include_sampler = None
    return_pdfs = None
    max_top_k_ids = None
    include_guided_decoding = None
    if qaic_config is not None:
        include_sampler = qaic_config.get("include_sampler", None)
        if include_sampler is not None:
            return_pdfs = qaic_config.get("return_pdfs", None)
            max_top_k_ids = qaic_config.get("max_top_k_ids", 512)
            include_guided_decoding = qaic_config.get("include_guided_decoding", None)

    #Check CCL is enabled
    if "ccl_enabled" in cfg or len(cfg.get("comp_ctx_lengths_prefill", [])) > 0 or len(cfg.get("comp_ctx_lengths_decode", [])) > 0:
        if qaic_config is None:
            qaic_config = dict()
        qaic_config["ccl_enabled"]=True
        if not cfg.pop("ccl_enabled", False):
            cfg["comp_ctx_lengths_prefill"] = [vllm_config.model_config.max_model_len] if (len(cfg.get("comp_ctx_lengths_prefill", [])) == 0) else cfg.get("comp_ctx_lengths_prefill", None)
            cfg["comp_ctx_lengths_decode"] = [vllm_config.model_config.max_model_len] if (len(cfg.get("comp_ctx_lengths_decode", [])) == 0) else cfg.get("comp_ctx_lengths_decode", None)

    qpc_path = cfg.pop("qpc_path")

    qpc_idx = None
    if kv_offload := cfg.pop("kv_offload", False):
        qpc_idx = (
            0 if vllm_config.model_config.runner_type == "pooling" else 1
        )
        if qpc_idx == 0:
            cfg["skip_lang"] = True
        else:
            cfg["skip_vision"] = True
    if qpc_path and kv_offload and len(qpc_path.split(":")) > 1:
        qpc_path = qpc_path.split(":")[qpc_idx]

    stages = int(cfg.pop("stages", 1))
    queue = cfg.pop("queue", None)
    cfg.pop("disable_multimodal", None)
    if vllm_config.kv_transfer_config and vllm_config.kv_transfer_config.kv_role == "kv_producer":
        # Only applicable for pipeline prefill
        stages = cfg['num_devices']

    if vllm_config.model_config.runner_type == "pooling" and not vllm_config.model_config.is_multimodal_model:
        if "prefill_seq_len" in cfg:
            cfg["seq_len"] = cfg["prefill_seq_len"]
            del cfg["prefill_seq_len"]
        if "ctx_len" in cfg:
            del cfg["ctx_len"]
        if "full_batch_size" in cfg:
            cfg["batch_size"] = cfg["full_batch_size"]
            del cfg["full_batch_size"]

    if "pooling_device" in cfg:
        del cfg["pooling_device"]
    if "pooling_method" in cfg:
        del cfg["pooling_method"]
    if "normalize" in cfg:
        del cfg["normalize"]
    if "softmax" in cfg:
        del cfg["softmax"]
    print(cfg)
    device_group =cfg.pop("device_group")

    if "io_encrypt" in cfg:
        # Currenlty Model IP flow is broken into two seperate runs
        # one to allow for the model to be compiled and then other to run using qpc route
        cfg["compile_only"] = True

    compile_only = cfg.pop("compile_only", False)

    # prefill_only option
    if vllm_config.kv_transfer_config:
        if "prefill_only" in cfg:
            assert ((kv_role == "kv_producer" and cfg["prefill_only"] == True) or
                    (kv_role == "kv_consumer" and cfg["prefill_only"] == False) or
                    (kv_role == "kv_both" and cfg["prefill_only"] == None)
                    ), ("prefill_only False is only supported for kv_consumer"
                    "prefill_only True is only supported for kv_producer"
                    "prefill_only None is only supported for kv_both")

    return QaicCompileConfig(
        compile_only=compile_only,
        qpc_path=qpc_path,
        device_group=device_group,
        cfg=cfg,
        num_logits_to_keep=num_logits_to_keep,
        kv_offload=kv_offload,
        qpc_idx=qpc_idx,
        include_sampler=include_sampler,
        include_guided_decoding=include_guided_decoding,
        return_pdfs=return_pdfs,
        max_top_k_ids=max_top_k_ids,
        qaic_config=qaic_config,
        stages=stages,
        queue=queue,
    )

def get_qaic_model(model: QaicCausalLM,
                   vllm_config: VllmConfig,
                   speculative_model_type: Optional[str] = None) -> nn.Module:

    if speculative_model_type is None:
        model.sampler.include_gpu_probs_tensor = False
    else:
        speculative_model_type = speculative_model_type.lower()
        model.sampler.include_gpu_probs_tensor =True

    if speculative_model_type not in QAIC_DEVICE_CONFIG:
        raise ValueError(
            f"Unable to find default profile for model type {speculative_model_type}!!\n")

    qaic_compile_config = _get_qaic_compile_config(vllm_config, speculative_model_type)
    qpc_path = qaic_compile_config.qpc_path

    # set lora max adapters
    if vllm_config.lora_config:
        qaic_max_adapters = int(os.environ.get("VLLM_QAIC_LORA_MAX_ID_SUPPORTED", 128))

    # if provided qpc is valid
    if qpc_path and not check_qpc_exists(qpc_path):
        raise ValueError(
            f"Environment variable VLLM_QAIC_QPC_PATH is set!\n"
            f"QAIC qpc path {qpc_path} doesn't exist or didn't have compiled binary!\n"
            "Unset VLLM_QAIC_QPC_PATH, if you don't want to provide compiled qpc.\n")

    # set adaptername_to_id from previous dump file if qpc_path exist
    adaptername_to_id = {}
    if vllm_config.lora_config and (qpc_path and check_qpc_exists(qpc_path)):
        # check if json file exist
        if os.path.exists(f'{qpc_path}/adaptername_to_id.json'):
            with open(f'{qpc_path}/adaptername_to_id.json', 'r') as file:
                adaptername_to_id = json.load(file)
        else:
            raise FileNotFoundError(f"The file at {qpc_path}/adaptername_to_id.json was not found. Please provide a correct VLLM_QAIC_QPC_PATH.")

        # check if json file content is correct
        if not verify_adaptername_to_id_consistency(adaptername_to_id, vllm_config.lora_config.lora_modules):
            raise ValueError(f"Inconsistent file content in {qpc_path}/adaptername_to_id.json and input lora modules.")

    # Generate qpc using QEfficient transformer
    if not qpc_path:
        import hashlib
        import requests
        from requests.exceptions import HTTPError

        # Check if model architecture is supported by QEfficient transformer
        #_check_model_architecture(model_config.hf_config)

        quant_cfg = vllm_config.model_config._parse_quant_hf_config()
        quant_method = None
        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

        if vllm_config.model_config.quantization is not None and \
            vllm_config.model_config.quantization in ["awq", "gptq"] and\
            quant_method!=vllm_config.model_config.quantization:
            raise ValueError(
                    f"Currently qaic backend only supports pre-quantized AWQ | GPTQ models"
                    " via vllm!"
                )

        try:
            qeff_model = get_hf_model(
                vllm_config.model_config,
                qaic_compile_config.qaic_config,
                vllm_config.lora_config,
                qaic_compile_config.kv_offload,
            )
            from QEfficient import QEFFAutoModelForCausalLM
            if not isinstance(qeff_model, QEFFAutoModelForCausalLM):
                # Only QEFFAutoModelForCausalLM supports the prefill-only option
                qaic_compile_config.cfg.pop("prefill_only", None)
            if vllm_config.lora_config:
                logger.info("Transforming and compiling lora model using QEfficient library")

                # search adapter in cache
                if not vllm_config.lora_config.lora_modules:

                    # search adapter in cache
                    filtered_cached_lora_module_paths = search_adapters_in_cache(vllm_config.model_config.model)

                    # error out if cache is empty
                    if len(filtered_cached_lora_module_paths) == 0:
                        raise ValueError("No adapter in cache, please either download some into HF_HOME or provide lora_modules list.")
                    # set lora_modules
                    vllm_config.lora_config.lora_modules = filtered_cached_lora_module_paths

                # error out if reach adapter limit
                assert len(vllm_config.lora_config.lora_modules) <= qaic_max_adapters, f"Number of cached adapters exceed limitation of {qaic_max_adapters}. Please either delete adapters from HF_HOME or specify adapters in lora_modules."

                # load adapters to model
                for lora_module_path in vllm_config.lora_config.lora_modules:
                    model_dir = lora_module_path.path.split('/')[-3]
                    adapter_model_id = f"{model_dir.split('--')[1]}/{model_dir.split('--')[2]}"
                    qeff_model.load_adapter(adapter_model_id, lora_module_path.name) # adapters with inconsistent target_modules or ranks will not be added here (TODO: otherwise should add another check here)

                # get adaptername_to_id
                adaptername_to_id = qeff_model.active_adapter_to_id

            else:
                logger.info(f"Transforming and compiling model[{speculative_model_type}] using QEfficient library")
            qeff_model.compile(**qaic_compile_config.cfg)
            qpc_path = qeff_model.qpc_path
            if isinstance(qpc_path, list):
                qpc_path = qpc_path[qaic_compile_config.qpc_idx]
        except Exception as e:
            logger.error("Failed to transform and compile the model! {e}")
            raise e

    # dump adaptername_to_id to folder for the first compilation
    if vllm_config.lora_config and not os.path.exists(f'{qpc_path}/adaptername_to_id.json'):
        with open(f'{qpc_path}/adaptername_to_id.json', 'w') as file:
            json.dump(adaptername_to_id, file)
            logger.info(f"Dump adaptername_to_id mapping to {qpc_path}/adaptername_to_id.json")

    if speculative_model_type != "default":
        logger.info(f"Spec model type {speculative_model_type}_{qaic_compile_config.num_logits_to_keep}")

    logger.info(f"Using qpc:-{qpc_path}")

    if qaic_compile_config.compile_only:
        # Hack for Model-IP execution flow
        # TODO: remove this in future
        # This will create error in parent process if exited,
        # need better solution in future
        logger.info("Compilation completed, exiting...")
        os.kill(os.getppid(), signal.SIGINT)
        time.sleep(10)
        exit(0)

    if vllm_config.model_config.is_multimodal_model:
        if vllm_config.model_config.hf_config.model_type != "internvl_chat":
            image_token_index = getattr(
                vllm_config.model_config.hf_config,
                "image_token_index",
                getattr(vllm_config.model_config.hf_config, "image_token_id", None)
            )
        else:
            from QEfficient.utils.constants import INTERN_3_5_IMG_CONTEXT_TOKEN, INTERN_IMG_CONTEXT_TOKEN
            image_token_index = (
                INTERN_3_5_IMG_CONTEXT_TOKEN
                if "Qwen3" in vllm_config.model_config.hf_config.architectures[0]
                else INTERN_IMG_CONTEXT_TOKEN
            )
    else:
        image_token_index = None

    # Load the weights from the cached or downloaded files.
    # model_config.qpc in None
    model.load_model(
        qpc_path=qpc_path,
        vocab_size=model.vocab_size,
        device_id=qaic_compile_config.device_group,
        seq_len=vllm_config.model_config.max_seq_len_to_capture,
        ctx_len=vllm_config.model_config.max_model_len,
        decode_bsz=vllm_config.scheduler_config.max_num_seqs,
        num_logits_to_keep=qaic_compile_config.num_logits_to_keep,
        speculative_model_type=speculative_model_type,
        lora_mode=bool(vllm_config.lora_config),
        include_sampler=qaic_compile_config.include_sampler,
        include_guided_decoding=qaic_compile_config.include_guided_decoding,
        return_pdfs=qaic_compile_config.return_pdfs,
        max_top_k_ids=qaic_compile_config.max_top_k_ids,
        is_multimodal_model=vllm_config.model_config.is_multimodal_model,
        stages=qaic_compile_config.stages,
        queue=qaic_compile_config.queue,
        kv_transfer_role=vllm_config.kv_transfer_config.kv_role if vllm_config.kv_transfer_config else None,
        image_token_index=image_token_index,
    )

    return model.eval()

def search_adapters_in_cache(base_model_name) -> List[str]:
    cached_lora_module_paths = []

    hf_home = os.environ.get("HF_HOME", None)
    hf_dir = f"{hf_home}/hub"
    if not any(os.scandir(hf_home)) or not any(os.scandir(hf_dir)):
        return []

    hf_root, hf_dirs, hf_files = next(os.walk(hf_dir))

    for model_dir in [m for m in hf_dirs if m.startswith("models")]: # walk through all models or adapter subdirs
        root, dirs, files = next(os.walk(f"{hf_root}/{model_dir}/snapshots"))
        if len(dirs) and os.path.isfile(f"{root}/{dirs[0]}/adapter_config.json"):
            cached_lora_module_paths.append(LoRAModulePath(name=f"{model_dir.split('--')[2]}", path=f"{root}/{dirs[0]}"))

    # filter out adapters that are with different base models
    filtered_cached_lora_module_paths = []
    for lora_module_path in cached_lora_module_paths:
        if PeftConfig.from_pretrained(lora_module_path.path).base_model_name_or_path == base_model_name:
            filtered_cached_lora_module_paths.append(lora_module_path)

    return filtered_cached_lora_module_paths

def verify_adaptername_to_id_consistency(json_file_input: Dict, lora_modules_input: List[LoRAModulePath]) -> bool:
    for i in range(len(lora_modules_input)):
        if (lora_modules_input[i].name not in json_file_input) or  (json_file_input[lora_modules_input[i].name] != i+1):
            return False
    return True

def update_hf_config(model_config: ModelConfig):
    # In QAic, num_patches is fixed,
    # so need to override max_dynamic_patch and min_dynamic_patch in hf_config
    if hasattr(model_config.hf_config, "max_dynamic_patch"):
        if model_config.override_qaic_config and "num_patches" in model_config.override_qaic_config:
            use_thumbnail = model_config.hf_config.use_thumbnail
            num_patches = int(model_config.override_qaic_config["num_patches"])
            model_config.hf_config.max_dynamic_patch = (
                num_patches - int(use_thumbnail)
            )
        model_config.hf_config.min_dynamic_patch = model_config.hf_config.max_dynamic_patch
