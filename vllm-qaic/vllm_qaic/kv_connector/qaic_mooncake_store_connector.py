# SPDX-License-Identifier: Apache-2.0
"""QAIC bridge for vLLM's MooncakeStoreConnector (cross-instance KV reuse).

Strategy (see qaic_kv_staging.py): keep a host CPU mirror of the on-card paged KV
block pool, register THAT with the stock MooncakeStoreConnector (so all of its
scheduler-side keying/lookup and worker-side put/get over the Mooncake store are
reused unchanged), and copy blocks card<->staging around the store I/O via the QAIC
session.

Status: the staging arena (the QAIC-coupled core) is implemented and CPU-unit-tested.
This connector wires it to the stock connector's lifecycle. The two box-side pieces
left are (a) deriving the exact block-id sets to move from the connector metadata,
and (b) ensuring the base get_finished/put path does not use CUDA events on the
CPU-only QAIC host. Until ``enable_qaic_transfers()`` is called (by the QAIC runner
once those are wired), this subclass behaves exactly like the base connector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm_qaic.kv_connector.qaic_kv_staging import QaicKVStagingArena

logger = init_logger(__name__)


class QaicMooncakeStoreConnector(MooncakeStoreConnector):
    """MooncakeStoreConnector that mirrors the on-card paged KV pool to host."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._arena: Optional["QaicKVStagingArena"] = None
        self._qaic_enabled: bool = False

    # ----- wiring from the QAIC model runner (worker role) -----
    def attach_staging_arena(self, arena: "QaicKVStagingArena") -> None:
        self._arena = arena

    def enable_qaic_transfers(self) -> None:
        """Turn on card<->staging copies (call once block-id wiring is validated).

        Fails fast (here, not mid-inference) if a subclass has not implemented the
        box-side block-id derivation.
        """
        if self._arena is None:
            raise RuntimeError("attach_staging_arena() must be called before enabling transfers")
        cls = type(self)
        if (
            cls._qaic_save_block_ids is QaicMooncakeStoreConnector._qaic_save_block_ids
            or cls._qaic_recv_block_ids is QaicMooncakeStoreConnector._qaic_recv_block_ids
        ):
            raise RuntimeError(
                "QAIC Mooncake block-id derivation is not implemented; a subclass must "
                "override _qaic_save_block_ids/_qaic_recv_block_ids before enabling transfers.")
        self._qaic_enabled = True

    # ----- worker-side lifecycle overrides -----
    def register_kv_caches(self, kv_caches: Optional[dict[str, "torch.Tensor"]] = None):
        # Register the HOST staging mirror (block pool) with Mooncake, not the
        # on-card tensors (which are not host-addressable).
        if self._arena is not None:
            return super().register_kv_caches(self._arena.as_kv_caches())
        if kv_caches is None:
            raise ValueError(
                "QaicMooncakeStoreConnector.register_kv_caches needs an attached staging "
                "arena (attach_staging_arena) or an explicit kv_caches dict.")
        return super().register_kv_caches(kv_caches)

    def wait_for_save(self):
        # Copy just-computed blocks card->staging so Mooncake puts fresh KV.
        if self._qaic_enabled and self._arena is not None:
            self._arena.card_to_staging(self._qaic_save_block_ids())
        return super().wait_for_save()

    def get_finished(self, finished_req_ids: set[str]):
        done_sending, done_recving = super().get_finished(finished_req_ids)
        # Drain blocks that Mooncake just received into the card for QPC decode.
        if self._qaic_enabled and self._arena is not None and done_recving:
            self._arena.staging_to_card(self._qaic_recv_block_ids(done_recving))
        return done_sending, done_recving

    # ----- box-side: derive physical block-id sets from connector metadata -----
    def _qaic_save_block_ids(self) -> Sequence[int]:
        # TODO(box): pull the physical block ids being saved this step from the
        # bound connector metadata (the same ids the base computes keys/addresses for).
        raise NotImplementedError("QAIC Mooncake save block-id wiring is box-side")

    def _qaic_recv_block_ids(self, recv_req_ids: set[str]) -> Sequence[int]:
        # TODO(box): map finished-recv request ids -> their physical block ids.
        raise NotImplementedError("QAIC Mooncake recv block-id wiring is box-side")
