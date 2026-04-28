# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""TurboConfig was adapted from vllm.transformers_utils.configs.medusa.py:MedusaConfig"""

import os
from typing import Union

from vllm.transformers_utils.configs.medusa import MedusaConfig

class TurboConfig(MedusaConfig):
    model_type = "turbo"
    config = "speculator_config.json"

    def __init__(self, *args, **kwargs):
        if "vocab_size" not in kwargs:
            kwargs["vocab_size"] = 128256
        if "architectures" not in kwargs:
            # set architecture to medusa to skip
            # registration of turbomodel for now
            kwargs["architectures"] = ["MedusaModel"]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "MedusaConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, _configuration_file=cls.config, **kwargs)
        for k in list(config_dict.keys()):
            if 'num' in k:
                if 'heads' in k:
                    config_dict["num_heads"] = config_dict.pop(k)
                elif 'layers' in k:
                    config_dict["num_hidden_layers"] = config_dict.pop(k)
        return cls.from_dict(config_dict, **kwargs)

    @property
    def n_predict(self):
        return self.num_heads