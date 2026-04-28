# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
from os.path import commonprefix
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from vllm.core.block.interfaces import Block, BlockAllocator, BlockId, Device, DeviceAwareBlockAllocator
from vllm.core.evictor import EvictionPolicy, Evictor, make_evictor
from vllm.sequence import Sequence
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator, assert_prefix_caching_block_or_none, PrefixCachingBlock, ComputedBlocksTracker
from vllm.core.block.block_table import BlockTable
SeqId = int
PrefixHash = int

class QaicEvictor(Evictor):
    """ qaic evictor wrapper class which manages different evictor for physical
    and virtual blocks
    """
    def __init__(self, no_physical_blk, eviction_policy = EvictionPolicy.LRU):
        self.no_physical_blk = no_physical_blk
        self.evictor_phy_blk: Evictor = make_evictor(eviction_policy)
        self.evictor_virt_blk: Evictor = make_evictor(eviction_policy)

    def __contains__(self, block_id: int) -> bool:
        if block_id < self.no_physical_blk:
            return self.evictor_phy_blk.__contains__(block_id)
        else:
            return self.evictor_virt_blk.__contains__(block_id)

    def evict(self, phy_blk_en = False) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        if phy_blk_en:
            blk_id,_cnt_hash = self.evictor_phy_blk.evict()
        else:
            blk_id,_cnt_hash = self.evictor_virt_blk.evict()

        return blk_id,_cnt_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""

        if block_id < self.no_physical_blk:
            self.evictor_phy_blk.add(block_id,
                                            content_hash,
                                            num_hashed_tokens,
                                            last_accessed)
        else:
            self.evictor_virt_blk.add(block_id,
                                            content_hash,
                                            num_hashed_tokens,
                                            last_accessed)

    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        if block_id < self.no_physical_blk:
            self.evictor_phy_blk.update(block_id, last_accessed)
        else:
            self.evictor_virt_blk.update(block_id, last_accessed)

    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        if block_id < self.no_physical_blk:
            self.evictor_phy_blk.remove(block_id)
        else:
            self.evictor_virt_blk.remove(block_id)

    def num_of_blocks(self, phy_blk_en = False) -> int:
        if phy_blk_en:
            return self.evictor_phy_blk.num_blocks
        else:
            return self.evictor_virt_blk.num_blocks
    @property
    def num_blocks(self) -> int:
        return self.evictor_phy_blk.num_blocks + self.evictor_virt_blk.num_blocks

class QaicPrefixCachedBlockDict():

    def __init__(self, no_physical_ids, refcounter, block_tracker)->None:
        self.cached_hash_virt = {}
        self.cached_hash_phy = {}
        self._cached_phy_id_table  = {}
        self._cached_virt_to_phy_id = {}
        self.block_tracker =block_tracker
        self._refcounter = refcounter
        self._no_physical_ids = no_physical_ids
        # Initial a empty Dict for every physical ID
        for blk_id in range(no_physical_ids):
            self._cached_phy_id_table[blk_id] = []

    def __contains__(self, key)->bool:
        return self.cached_hash_virt.__contains__(key) or self.cached_hash_phy.__contains__(key)

    def cache_hit_if_any(self, block: PrefixCachingBlock)->Optional[int]:

        if not self.__contains__(block.content_hash):
            return None

        if block.prev_block is None:
            blk_ids_possible = self.cached_hash_phy[block.content_hash]
            #LRU pop implementation
            _last_used = float("inf")
            _last_phy_id = None
            for _blk_id in blk_ids_possible:
                if self._refcounter.get(_blk_id) == 0:
                    if len(self._cached_phy_id_table[_blk_id]) == 0:
                        _last_accessed = self.block_tracker[_blk_id].last_accessed
                    else:
                        _last_accessed = self.block_tracker[self._cached_phy_id_table[_blk_id][-1][0]].last_accessed

                    if _last_used > _last_accessed:
                        _last_used = _last_accessed
                        _last_phy_id = _blk_id
            return _last_phy_id

        else:
            phy_id = block.prev_block._phy_blk_id
            cnt_hash = block.content_hash
            for (_virt_id, _cnt_hash) in self._cached_phy_id_table[phy_id]:
                if _cnt_hash == cnt_hash and self._refcounter.get(_virt_id) == 0:
                    return _virt_id
            return None

    def add(self, block, block_size)->None:
        if block.prev_block is None:
            _phy_id = block.block_id

            assert len(self._cached_phy_id_table[_phy_id]) == 0
            _cnt_hash = block.content_hash
            if _cnt_hash in self.cached_hash_phy:
                self.cached_hash_phy[_cnt_hash].add(_phy_id)
            else:
                self.cached_hash_phy[_cnt_hash] = set([_phy_id])

            return None
        else:
            _phy_id = block.prev_block._phy_blk_id
            virt_id = block.block_id
            cnt_hash = block.content_hash
            evict_list = []
            blk_idx = int((block.num_tokens_total / block_size)) -1

            if len(self._cached_phy_id_table[_phy_id]) >= blk_idx:
                for _virt_id, _cnt_hash in self._cached_phy_id_table[_phy_id][blk_idx-1:]:
                    if _virt_id != virt_id:
                        evict_list.append(_virt_id)
                    if _cnt_hash in self.cached_hash_virt:
                        self.remove_hash(self.cached_hash_virt, _virt_id, _cnt_hash)

                        del self._cached_virt_to_phy_id[_virt_id]
                self._cached_phy_id_table[_phy_id] = self._cached_phy_id_table[_phy_id][:blk_idx-1]

            self._cached_phy_id_table[_phy_id].append((virt_id, cnt_hash))
            if cnt_hash in self.cached_hash_virt:
                self.cached_hash_virt[cnt_hash].add(virt_id)
            else:
                self.cached_hash_virt[cnt_hash] = set([virt_id])

            self._cached_virt_to_phy_id[virt_id] = (_phy_id, len( self._cached_phy_id_table[_phy_id])-1)

            return evict_list

    def remove_hash(self, cached_dict, blk_id, cont_hash):
        assert blk_id in cached_dict[cont_hash]
        if len(cached_dict[cont_hash]) == 1:
            cached_dict.pop(cont_hash)
        else:
            cached_dict[cont_hash].remove(blk_id)

    def remove_and_get_evicted_virtual_ids(self, blk_id: BlockId, phy_blk_en: bool,  content_hash_to_evict:PrefixHash)->List:

        if not phy_blk_en:

            phy_blk_id, idx = self._cached_virt_to_phy_id[blk_id]
        else:
            phy_blk_id, idx = blk_id, 0
            self.remove_hash(self.cached_hash_phy, phy_blk_id, content_hash_to_evict)

        max_virt_ids = len(self._cached_phy_id_table[phy_blk_id])
        result = []
        for _virt_id,_cnt_hash in self._cached_phy_id_table[phy_blk_id][-max_virt_ids + idx:]:
            result.append(_virt_id)
            self.remove_hash(self.cached_hash_virt, _virt_id, _cnt_hash)
            del self._cached_virt_to_phy_id[_virt_id]
            self._cached_phy_id_table[phy_blk_id].pop()

        return result

class QaicPrefixCachingBlockAllocator(PrefixCachingBlockAllocator):

    def __init__(self, num_blocks, num_phy_blocks, block_size, block_ids = None, eviction_policy = EvictionPolicy.LRU):
        super().__init__(num_blocks, block_size, block_ids, eviction_policy)
        self._no_physical_blk: int = num_phy_blocks
        self._free_physical_blk: Deque[BlockId] = deque(range(self._no_physical_blk))
        self._frozen_physical_id_set = frozenset(self._free_physical_blk)

        #Allocate physical IDs
        for id in self._frozen_physical_id_set:
            self._hashless_allocator._free_block_indices.remove(id)

        # qaic sw infra currently don't supports KV cache block sharing
        # So the whole sequence uses one big block instead of multiple blks
        self._cached_blocks: QaicPrefixCachedBlockDict = QaicPrefixCachedBlockDict(self._no_physical_blk, self._refcounter, self._block_tracker)
        self.evictor = QaicEvictor(self._no_physical_blk, eviction_policy)

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 extra_hash: Optional[int] = None,
                                 device: Optional[Device] = None) -> Block:
        """Allocates an immutable block with the given token IDs, reusing cached
        blocks if possible.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
            token_ids (List[int]): The token IDs to be stored in the block.

        Returns:
            Block: The allocated immutable block.
        """
        assert device is None
        assert_prefix_caching_block_or_none(prev_block)

        # First, try to create a block that points to cached data
        block = self._block_pool.init_block(prev_block=prev_block,
                                            token_ids=token_ids,
                                            block_size=self._block_size,
                                            physical_block_id=None,
                                            extra_hash=extra_hash)
        assert block.content_hash is not None
        cached_block_id = self._cached_blocks.cache_hit_if_any(block)
        if cached_block_id is not None:
            self.metric_data.query(hit=True)
            block.block_id = cached_block_id
            assert block.block_id not in self._free_physical_blk
            self._incr_refcount_cached_block(block)
            self._update_phy_id(block, prev_block)

            return block

        self.metric_data.query(hit=False)
        self._block_pool.free_block(block)

        # No cached block => Allocate a new block
        block = self.allocate_mutable_block(prev_block, extra_hash=extra_hash)
        block.append_token_ids(token_ids)
        return block

    def _update_phy_id(self, block: Block, prev_block: Optional[Block]):
        if prev_block is None:
            block._phy_blk_id =  block.block_id
        else:
            block._phy_blk_id = prev_block._phy_blk_id

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        """Once a mutable block is full, it can be promoted to an immutable
        block. This means that its content can be referenced by future blocks
        having the same prefix.

        Note that if we already have a cached block with the same content, we
        will replace the newly-promoted block's mapping with the existing cached
        block id.

        Args:
            block: The mutable block to be promoted.

        Returns:
            BlockId: Either the original block index, or the block index of
                the previously cached block matching the same content.
        """
        # Ensure block can be promoted
        assert block.content_hash is not None
        assert block.block_id is not None
        assert self._refcounter.get(block.block_id) > 0

        anything_to_evict = self._cached_blocks.add(block, self._block_size)

        if anything_to_evict is not None:
            for _virt_id in  anything_to_evict:

                assert self._refcounter.get(_virt_id) == 0
                self.evictor.evictor_virt_blk.remove(_virt_id)
                self._hashless_allocator._free_block_indices.appendleft(_virt_id)

        self._touched_blocks.add(block.block_id)

        return block.block_id

        raise BlockAllocator.NoFreeBlocksError()

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               extra_hash: Optional[int] = None,
                               device: Optional[Device] = None) -> Block:
        """Allocates a mutable block. If there are no free blocks, this will
        evict unused cached blocks.

        Args:
            prev_block (Block): The previous block in the sequence.
                None is not allowed unlike it is super class.

        Returns:
            Block: The allocated mutable block.
        """
        assert device is None
        assert_prefix_caching_block_or_none(prev_block)

        block_id = self._allocate_block_id(prev_block)
        block = self._block_pool.init_block(prev_block=prev_block,
                                            token_ids=[],
                                            block_size=self._block_size,
                                            physical_block_id=block_id,
                                            extra_hash=extra_hash)
        assert not block.computed
        assert block.content_hash is None
        self._update_phy_id(block, prev_block)
        return block

    def _allocate_block_id(self, prev_block: Optional[Block]) -> BlockId:
        """First tries to allocate a block id from the hashless allocator,
        and if there are no blocks, then tries to evict an unused cached block.
        """
        # If this is first block in the chain, allocate physical address block
        if prev_block is None:
            # Allocate a virtual id which is same as physical ID
            #as modelrunner only propogate block id of first entry in
            #block table
            if len(self._free_physical_blk) > 0:
                block_id = self._free_physical_blk.popleft()
                assert self._refcounter.get(block_id) == 0
                #allocate the blockid in hashless_allocator
                self._refcounter.incr(block_id)
                self._track_block_id(block_id, computed=False)
                return block_id
            else:
                evicted_block_id = self._maybe_allocate_evicted_block_id(True)
                if evicted_block_id is not None:
                    return evicted_block_id
        else:
            hashless_block_id = self._maybe_allocate_hashless_block_id()
            if hashless_block_id is not None:
                return hashless_block_id

            evicted_block_id = self._maybe_allocate_evicted_block_id(False)
            if evicted_block_id is not None:
                return evicted_block_id

        # No block available in hashless allocator, nor in unused cache blocks.
        raise BlockAllocator.NoFreeBlocksError()

    def _maybe_allocate_evicted_block_id(self, phy_blk_en) -> Optional[BlockId]:
        if self.evictor.num_of_blocks(phy_blk_en) == 0:
            return None

        # Here we get an evicted block, which is only added
        # into evictor if its ref counter is 0
        # and since its content would be changed, we need
        # to remove it from _cached_blocks's tracking list
        block_id, content_hash_to_evict = self.evictor.evict(phy_blk_en)

        # Sanity checks
        assert content_hash_to_evict in self._cached_blocks
        assert self._refcounter.get(block_id) == 0

        blk_ids = self._cached_blocks.remove_and_get_evicted_virtual_ids(block_id, phy_blk_en, content_hash_to_evict)
        for _blk_id in blk_ids:
            assert self._refcounter.get(_blk_id) == 0
            if _blk_id != block_id:
                self.evictor.evictor_virt_blk.remove(_blk_id)
            self._hashless_allocator._free_block_indices.appendleft(_blk_id)

        self._refcounter.incr(block_id)
        self._track_block_id(block_id, computed=False)

        return block_id

    def _decr_refcount_hashless_block(self, block: Block) -> None:
        block_id = block.block_id
        assert block_id is not None

        # We may have a fork case where block is shared,
        # in which case, we cannot remove it from tracking
        refcount = self._refcounter.get(block_id)
        if refcount == 1:
            self._untrack_block_id(block_id)

        # Decrement refcount of the block_id, but do not free the block object
        # itself (will be handled by the caller)
        # dont free physical blocks
        if block_id not in self._frozen_physical_id_set:
            self._hashless_allocator.free(block, keep_block_object=True)
        else:
            refcount = self._refcounter.decr(block_id)
            assert refcount == 0
            self._free_physical_blk.appendleft(block_id)
            assert block.block_id not in self.evictor
            block.block_id = None

    def find_cached_blocks_prefix(self, block_hashes: List[int], all_blocks:Optional[List[int]] = None) -> List[int]:
        """
        Given a list of block hashes, return the prefix of the block hashes that
        are all cached.

        Since a block's block hash includes the hashes of all previous blocks,
        and we only allocate/deallocate blocks in the entire sequence, so if a
        block is cached, then all previous blocks are also cached. With this
        property, we can use binary search to find the prefix of cached blocks.

        Args:
            block_hashes (List[int]): The list of block hashes.

        Returns:
            List[int]: The prefix of the `block_hashes` that are cached.
        """
        if len(block_hashes) == 0 or block_hashes[0] not in self._cached_blocks.cached_hash_phy:
            return []

        phy_blk_set = self._cached_blocks.cached_hash_phy[block_hashes[0]]

        if all_blocks == None:
            idx = len(block_hashes)
            max_len = idx
            i = 0
            # Count how many blocks are already cached
            for _blk_id in phy_blk_set:
                if not self.block_is_computed(_blk_id):
                    continue
                else:
                    i = 1
                    for _virt_id, _cnt_hash in self._cached_blocks._cached_phy_id_table[_blk_id]:
                        if idx == max_len or block_hashes[idx] != _cnt_hash or not self.block_is_computed(_virt_id):
                            break
                        i +=1
                    idx = min(idx, i)
        else:
            max_len = len(self._cached_blocks._cached_phy_id_table[all_blocks[0]])

            assert all_blocks[0] in phy_blk_set
            idx = 0

            for blk_id in all_blocks:
                if idx == 0:
                    _virt_id = blk_id
                    _cnt_hash = block_hashes[0]
                elif (idx-1) != max_len:
                    (_virt_id, _cnt_hash) = self._cached_blocks._cached_phy_id_table[all_blocks[0]][idx-1]

                if ((idx-1) == max_len or 
                    blk_id != _virt_id or 
                    _cnt_hash != block_hashes[idx] or 
                    not self.block_is_computed(_virt_id)):
                    break
                idx +=1

        return block_hashes[:idx] if idx > 0 else []

    def get_num_full_blocks_touched(self, blocks: List[Block]) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out.

        Args:
            blocks: List of blocks to be swapped.
        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks. Non full blocks are ignored
                when deciding the number of blocks to touch.
        """
        num_touched_blocks: int = 0
        for block in blocks:
            # If the block has a match in the cache and the cached
            # block is not referenced, then we still count it as a
            # touched block
            if block.is_full and (not self.is_block_cached(block) or \
                (block.content_hash is not None and \
                self._cached_blocks[block.content_hash] in \
                        self.evictor)):
                num_touched_blocks += 1
        return num_touched_blocks

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        """Return the block ids that are common for a given sequence group.

        Only those blocks that are immutable and already be marked
        compyted would be taken consideration.
        """
        # NOTE We exclude the last block to avoid the case where the entire
        # prompt is cached. This would cause erroneous behavior in model
        # runner.

        # It returns a list of int although type annotation says list of string.
        if len(computed_seq_block_ids) == 1:
            return computed_seq_block_ids[0]

        return commonprefix([
            ids for ids in computed_seq_block_ids  # type: ignore
            if ids
        ])

class QaicComputedBlocksTracker(ComputedBlocksTracker):
    def __init__(
        self,
        allocator: DeviceAwareBlockAllocator,
        block_size: int,
        enable_caching: bool,
        block_tables: Dict[SeqId, BlockTable] = {}
    ):
        super().__init__(allocator, block_size, enable_caching)
        self.block_tables = block_tables

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        if not self._enable_caching:
            return 0

        # We always try to update the sequence hashes on the fly.
        # This is to ensure that we don't miss any cached tokens for the
        # sequence during decode.
        # This routine should only update hash for any new blocks too.
        self._update_seq_hashes(seq)

        num_computed_tokens_prev = self._seq_id_to_num_tokens_computed.get(
            seq.seq_id, None)

        try:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
        except:
            all_blocks= None
        # TODO(rickyx): This hack could be removed once we mark blocks as
        # computed correctly with chunked prefills.
        # if num_computed_tokens_prev is not None and seq.is_prefill():
        #     # For a sequence that is still in prefill, we don't
        #     # recompute the number of cached tokens.
        #     # This also handles correctly chunked prefill since currently
        #     # we mark blocks as computed even if the sequence is still partially
        #     # prefilled. So a continuously prefilled sequence should not
        #     # see its cached token count change while running.
        #     return num_computed_tokens_prev

        # Hack, assuming we don't support chunked prefill at scheduler level yet
        #if not seq.is_prefill() and num_computed_tokens_prev is not None:
        #    return num_computed_tokens_prev

        block_hashes = self._seq_id_to_blocks_hashes[seq.seq_id]


        # This is O(logN), where N is the number of blocks.
        # In QAIC multiple hashes are not unique in the cache, multiple block id can have same
        # Block hash, so block ids are needed to resolve the conflict
        num_cached_blocks = len(
            self._allocator._allocators[Device.GPU].find_cached_blocks_prefix(block_hashes, all_blocks))
        num_cached_tokens = num_cached_blocks * self._block_size
        self._seq_id_to_num_tokens_computed[seq.seq_id] = num_cached_tokens

        return num_cached_tokens