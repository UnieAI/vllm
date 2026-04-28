# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Not a contribution.
# ---------------------------------------------------------------------------------------
# Temporarily reverting PR #21785 [V0 deprecation][P/D] Deprecate v0 KVConnectorBase code (1/2)
# for backward compatibility with v0.
import importlib
from typing import TYPE_CHECKING, Callable

# yapf: disable
import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase, KVConnectorBaseType)
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.logger import init_logger

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBaseType]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBaseType]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector_v0(cls, rank: int, local_rank: int,
                            config: "VllmConfig") -> KVConnectorBase:
        if envs.VLLM_USE_V1:
            raise ValueError("Attempting to initialize a V0 Connector, "
                             f"but found {envs.VLLM_USE_V1=}")

        connector_cls = cls.get_connector_class(config.kv_transfer_config)
        assert issubclass(connector_cls, KVConnectorBase)
        return connector_cls(rank, local_rank, config)

    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
    ) -> KVConnectorBase:
        if not envs.VLLM_USE_V1:
            raise ValueError("Attempting to initialize a V1 Connector, "
                             f"but found {envs.VLLM_USE_V1=}")

        kv_transfer_config = config.kv_transfer_config
        connector_cls = cls.get_connector_class(kv_transfer_config)
        logger.info("Creating v1 connector with name: %s and engine_id: %s",
                    connector_cls.__name__, kv_transfer_config.engine_id)
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        return connector_cls(config, role)

    @classmethod
    def get_connector_class(
            cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        """Get the connector class by name."""
        connector_name = kv_transfer_config.kv_connector
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            connector_module_path = kv_transfer_config.kv_connector_module_path
            if connector_module_path is None:
                raise ValueError(
                    f"Unsupported connector type: {connector_name}")
            connector_module = importlib.import_module(connector_module_path)
            connector_cls = getattr(connector_module, connector_name)
        return connector_cls


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

KVConnectorFactory.register_connector(
    "SharedStorageConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
    "SharedStorageConnector")

KVConnectorFactory.register_connector(
    "P2pNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector",
    "P2pNcclConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1")

KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector",
    "NixlConnector")

KVConnectorFactory.register_connector(
    "MultiConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
    "MultiConnector")

KVConnectorFactory.register_connector(
    "QaicConnector",
    "vllm.distributed.kv_transfer.kv_connector.qaic_connector",
    "QaicConnector")
