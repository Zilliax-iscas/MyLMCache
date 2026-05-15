# SPDX-License-Identifier: Apache-2.0
"""Request-scoped decode-overlap KV prefetch scaffolding.

This module intentionally contains only orchestration and mock transfer hooks.
It models the target pipeline:

1. Scheduler sees a waiting request with LMCache hits and records a prefetch
   intent.
2. A background worker will eventually stage hit KV from CPU storage to a GPU
   staging buffer while another request is decoding.
3. When the request is scheduled, the worker consumes the staged handle and
   copies it into vLLM KV slots before blending/recompute.

The current implementation marks mock prefetches ready immediately and falls
back to the normal retrieve path. Real GPU staging can replace the mock methods
without changing the adapter call sites.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import deque
import hashlib
import json
import os
from threading import Lock
import threading
import time
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata

logger = init_logger(__name__)


@dataclass
class DecodeOverlapPrefetchIntent:
    """Metadata for one request's planned decode-overlap prefetch."""

    req_id: str
    token_count: int
    lmcache_hit_tokens: int
    vllm_cached_tokens: int
    request_configs: Optional[dict[str, Any]] = None
    token_signature: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    ready_at: Optional[float] = None
    consumed_at: Optional[float] = None
    state: str = "pending"
    note: str = ""


@dataclass
class DecodeOverlapPrefetchHandle:
    """A consumed prefetch result.

    Future fields should point at GPU staging buffers and CUDA events.  For now
    this is a mock handle that proves the scheduler-to-worker lifecycle.
    """

    req_id: str
    token_count: int
    lmcache_hit_tokens: int
    vllm_cached_tokens: int
    ready_at: Optional[float]
    request_configs: Optional[dict[str, Any]] = None
    token_signature: Optional[str] = None
    mock_only: bool = True
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    direct_final_slot: bool = False
    staged_layers: list[list["StagedMemoryObj"]] = field(default_factory=list)
    source_layers: list[list[MemoryObj]] = field(default_factory=list)
    layer_ready_events: list[Optional[torch.cuda.Event]] = field(default_factory=list)
    ready_event: Optional[torch.cuda.Event] = None
    _waited: bool = False
    _released_sources: bool = False
    _released_source_layer_ids: set[int] = field(default_factory=set)
    _layer_cond: threading.Condition = field(
        default_factory=threading.Condition,
        repr=False,
    )

    def can_supply_layerwise(
        self,
        starts: list[int],
        ends: list[int],
        num_layers: int,
    ) -> bool:
        return (
            not self.mock_only
            and self.starts == starts
            and self.ends == ends
            and len(self.staged_layers) == num_layers
            and self.ready_layer_count() > 0
        )

    def ready_layer_count(self) -> int:
        """Return the number of contiguous staged layers from layer 0."""

        with self._layer_cond:
            count = 0
            for staged_layer in self.staged_layers:
                if not staged_layer:
                    break
                count += 1
            return count

    def get_layer(self, layer_id: int) -> list["StagedMemoryObj"]:
        with self._layer_cond:
            while not self.staged_layers[layer_id]:
                self._layer_cond.wait(timeout=0.001)
            layer = self.staged_layers[layer_id]
            ready_event = (
                self.layer_ready_events[layer_id]
                if layer_id < len(self.layer_ready_events)
                else None
            )

        if ready_event is not None:
            ready_event.synchronize()
        self.release_source_layer_refs(layer_id)
        return layer

    def add_layer(
        self,
        layer_id: int,
        staged_layer: list["StagedMemoryObj"],
        source_layer: list[MemoryObj],
        ready_event: Optional[torch.cuda.Event],
    ) -> None:
        with self._layer_cond:
            self.staged_layers[layer_id] = staged_layer
            self.source_layers[layer_id] = source_layer
            self.layer_ready_events[layer_id] = ready_event
            self._layer_cond.notify_all()

    def can_supply_final_slots(
        self,
        starts: list[int],
        ends: list[int],
        num_layers: int,
    ) -> bool:
        return (
            not self.mock_only
            and self.direct_final_slot
            and self.starts == starts
            and self.ends == ends
            and len(self.source_layers) == num_layers
        )

    def get_source_layer(self, layer_id: int) -> list[MemoryObj]:
        self.wait_ready()
        return self.source_layers[layer_id]

    def wait_ready(self) -> None:
        if self._waited:
            return
        if self.ready_event is not None:
            self.ready_event.synchronize()
        self._waited = True

    def release_source_refs(self) -> None:
        if self._released_sources:
            return
        for layer_id in range(len(self.source_layers)):
            self.release_source_layer_refs(layer_id)
        self._released_sources = True

    def release_source_layer_refs(self, layer_id: int) -> None:
        if layer_id in self._released_source_layer_ids:
            return
        if layer_id >= len(self.source_layers):
            return
        for mem_obj in self.source_layers[layer_id]:
            mem_obj.ref_count_down()
        self._released_source_layer_ids.add(layer_id)


@dataclass
class StagedMemoryObj:
    """Small MemoryObj-like wrapper around a GPU staged tensor."""

    tensor_value: torch.Tensor
    metadata: MemoryObjMetadata

    @property
    def tensor(self) -> torch.Tensor:
        return self.tensor_value

    @property
    def is_pinned(self) -> bool:
        return False

    def ref_count_down(self) -> None:
        return None

    def unpin(self) -> bool:
        return True


class DecodeOverlapPrefetchCoordinator:
    """Tracks request prefetch intents and exposes mock transfer hooks."""

    def __init__(self, name: str, max_entries: int = 128):
        self.name = name
        self.max_entries = max_entries
        self._intents: dict[str, DecodeOverlapPrefetchIntent] = {}
        self._handles: dict[str, DecodeOverlapPrefetchHandle] = {}
        self._lock = Lock()
        self._engine: Optional[Any] = None
        self._token_aliases: dict[str, str] = {}
        self._req_aliases: dict[str, str] = {}
        self._proactive_hint_ids: set[str] = set()
        self._hint_watcher_started = False
        self._pending_proactive: deque[
            tuple[str, list[int], Optional[dict[str, Any]]]
        ] = deque()
        self._decode_cond = threading.Condition()
        self._decode_active = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{name}-decode-overlap-prefetch",
        )

    def register_worker_engine(self, engine: Any) -> None:
        """Attach the worker-side LMCacheEngine used for real staging."""

        self._engine = engine
        logger.info("Decode-overlap prefetch registered worker engine: %s", self.name)
        self._start_hint_watcher_if_configured()

    def begin_decode_step(self, reason: str = "") -> None:
        """Open a decode-only window for proactive H2D prefetch.

        This should be called from the vLLM worker immediately before a decode
        forward step.  The prefetch worker is allowed to submit and continue
        H2D copies only while this window is open.
        """

        with self._decode_cond:
            self._decode_active = True
            self._decode_cond.notify_all()
        self._drain_pending_proactive()
        logger.debug("Decode-overlap decode window opened: %s", reason)

    def end_decode_step(self, reason: str = "") -> None:
        """Close the decode-only prefetch window."""

        with self._decode_cond:
            self._decode_active = False
            self._decode_cond.notify_all()
        logger.debug("Decode-overlap decode window closed: %s", reason)

    def _wait_for_decode_window(self) -> bool:
        """Block proactive staging until vLLM is inside a decode step."""

        if not self._decode_window_required():
            return True
        with self._decode_cond:
            while not self._decode_active:
                self._decode_cond.wait(timeout=0.001)
            return True

    def _decode_window_required(self) -> bool:
        return os.environ.get(
            "LMCACHE_DECODE_OVERLAP_REQUIRE_DECODE_WINDOW",
            "True",
        ).lower() in {"1", "true", "yes", "on"}

    def submit_intent(
        self,
        req_id: str,
        token_count: int,
        lmcache_hit_tokens: int,
        vllm_cached_tokens: int,
        tokens: Optional[list[int]] = None,
        mask: Optional[torch.Tensor] = None,
        request_configs: Optional[dict[str, Any]] = None,
        note: str = "",
    ) -> DecodeOverlapPrefetchIntent:
        """Record that a request is eligible for decode-overlap prefetch."""

        token_signature = self._token_signature(tokens) if tokens is not None else None
        is_proactive = req_id.startswith("proactive:") or note == "proactive_hint_file"
        with self._lock:
            if token_signature is not None and not is_proactive:
                alias_req_id = self._token_aliases.get(token_signature)
                if alias_req_id is not None and alias_req_id != req_id:
                    self._req_aliases[req_id] = alias_req_id
                    intent = self._intents.get(alias_req_id)
                    if intent is not None:
                        logger.info(
                            "Decode-overlap bound req_id=%s to proactive "
                            "prefetch=%s by token signature.",
                            req_id,
                            alias_req_id,
                        )
                        return intent

            self._evict_oldest_locked()
            intent = DecodeOverlapPrefetchIntent(
                req_id=req_id,
                token_count=token_count,
                lmcache_hit_tokens=lmcache_hit_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                request_configs=request_configs,
                token_signature=token_signature,
                note=note,
            )
            self._intents[req_id] = intent
            if token_signature is not None and is_proactive:
                self._token_aliases[token_signature] = req_id

        should_stage = is_proactive or self._stage_passive_enabled()
        if (
            should_stage
            and tokens is not None
            and mask is not None
            and self._engine is not None
        ):
            self._executor.submit(
                self._stage_cpu_to_gpu_safe,
                req_id,
                tokens,
                mask,
                request_configs,
            )
        elif should_stage:
            self.mock_stage_cpu_to_gpu(req_id)
        else:
            logger.debug(
                "Decode-overlap passive intent recorded without staging: req_id=%s",
                req_id,
            )
        logger.info(
            "Decode-overlap prefetch intent submitted: req_id=%s, "
            "token_count=%d, lmcache_hit_tokens=%d, vllm_cached_tokens=%d",
            req_id,
            token_count,
            lmcache_hit_tokens,
            vllm_cached_tokens,
        )
        return intent

    def submit_rag_prefetch(
        self,
        *,
        hint_id: str,
        tokens: list[int],
        request_configs: Optional[dict[str, Any]] = None,
    ) -> DecodeOverlapPrefetchIntent:
        """Submit a RAG-owned proactive prefetch request.

        This is the public shape we want from a real RAG service: the RAG layer
        already knows the next request's retrieved chunks, represented here by
        the full post-retrieval prompt tokens. The coordinator performs the
        LMCache lookup, stages CPU KV into GPU compute buffers, and later binds
        the real vLLM request by token signature.

        A future chunk-id API can replace `tokens` without changing the
        lifecycle: RAG submit -> CPU_READY lookup -> GPU prefetch -> consume.
        """

        req_id = f"proactive:{hint_id}"
        if self._decode_window_required():
            with self._decode_cond:
                decode_active = self._decode_active
            if not decode_active:
                with self._lock:
                    self._pending_proactive.append((hint_id, tokens, request_configs))
                logger.info(
                    "Decode-overlap queued proactive prefetch until decode "
                    "window: req_id=%s, token_count=%d",
                    req_id,
                    len(tokens),
                )
                return DecodeOverlapPrefetchIntent(
                    req_id=req_id,
                    token_count=len(tokens),
                    lmcache_hit_tokens=len(tokens),
                    vllm_cached_tokens=0,
                    request_configs=request_configs,
                    token_signature=self._token_signature(tokens),
                    note="queued_until_decode",
                )

        chunk_size = int(os.environ.get("LMCACHE_CHUNK_SIZE", "256"))
        aligned_token_count = len(tokens) // chunk_size * chunk_size
        mask = torch.zeros(len(tokens), dtype=torch.bool)
        if aligned_token_count > 0:
            mask[:aligned_token_count] = True
        logger.info(
            "Decode-overlap proactive prefetch aligned to full chunks: "
            "req_id=%s, prompt_tokens=%d, prefetch_tokens=%d, chunk_size=%d",
            req_id,
            len(tokens),
            aligned_token_count,
            chunk_size,
        )
        return self.submit_intent(
            req_id=req_id,
            token_count=len(tokens),
            lmcache_hit_tokens=aligned_token_count,
            vllm_cached_tokens=0,
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
            note="proactive_hint_file",
        )

    def _stage_cpu_to_gpu_safe(
        self,
        req_id: str,
        tokens: list[int],
        mask: torch.Tensor,
        request_configs: Optional[dict[str, Any]],
    ) -> None:
        try:
            self.stage_cpu_to_gpu(
                req_id=req_id,
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            )
        except Exception:
            logger.exception(
                "Decode-overlap real prefetch failed for req_id=%s; "
                "falling back to normal retrieve.",
                req_id,
            )
            self.cancel(req_id, reason="real prefetch failed")

    def mock_stage_cpu_to_gpu(self, req_id: str) -> None:
        """Mock CPU->GPU staging.

        Real implementation should allocate a GPU staging buffer, issue async
        copies on a separate CUDA stream, and store the CUDA event here.
        """

        with self._lock:
            intent = self._intents.get(req_id)
            if intent is None or intent.state in {"cancelled", "consumed"}:
                return
            intent.state = "ready"
            intent.ready_at = time.time()

        logger.info(
            "Decode-overlap mock prefetch ready: req_id=%s. "
            "Real CPU->GPU staging is not implemented yet.",
            req_id,
        )

    def stage_cpu_to_gpu(
        self,
        req_id: str,
        tokens: list[int],
        mask: torch.Tensor,
        request_configs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fetch hit KV objects and stage them in GPU compute buffers.

        The staged GPU buffers are shaped exactly like the layerwise blend
        compute buffer ([2, num_tokens, hidden_dim]).  When the request is later
        scheduled, retrieve_layer can attach these buffers directly to the GPU
        connector's buffer_mapping so CacheBlend recompute reads prefetched KV
        without doing CPU->GPU H2D on the request critical path.
        """

        engine = self._engine
        if engine is None or engine.storage_manager is None:
            raise RuntimeError("Worker LMCacheEngine is not registered for prefetch")
        starts, ends, keys_layer_major, location = self._build_layerwise_plan(
            engine,
            tokens,
            mask,
            request_configs,
        )
        if not keys_layer_major:
            self.cancel(req_id, reason="no layerwise keys to prefetch")
            return

        get_generator = engine.storage_manager.layerwise_batched_get(
            keys_layer_major,
            location=location,
        )

        planned_signature = self._token_signature(tokens[: ends[-1]])
        source_layers: list[list[MemoryObj]] = [[] for _ in range(engine.num_layers)]
        staged_layers: list[list[StagedMemoryObj]] = [
            [] for _ in range(engine.num_layers)
        ]
        layer_ready_events: list[Optional[torch.cuda.Event]] = [
            None for _ in range(engine.num_layers)
        ]
        initial_layer_count = self._initial_ready_layer_count(
            req_id,
            engine.num_layers,
        )
        handle: Optional[DecodeOverlapPrefetchHandle] = None
        is_proactive = req_id.startswith("proactive:")

        for layer_id in range(engine.num_layers):
            task = next(get_generator)
            assert task is not None
            source_layer = task.result()
            staged_layer, ready_event = self._stage_one_layer_to_gpu_buffer(
                engine,
                starts,
                ends,
                source_layer,
            )
            source_layers[layer_id] = source_layer
            staged_layers[layer_id] = staged_layer
            layer_ready_events[layer_id] = ready_event

            if handle is None and layer_id + 1 >= initial_layer_count:
                with self._lock:
                    intent = self._intents.get(req_id)
                    if intent is None:
                        for release_layer in source_layers:
                            for mem_obj in release_layer:
                                mem_obj.ref_count_down()
                        return
                    intent.state = "ready"
                    intent.ready_at = time.time()
                    intent.token_signature = planned_signature
                    if (
                        req_id.startswith("proactive:")
                        or intent.note == "proactive_hint_file"
                    ):
                        self._token_aliases[planned_signature] = req_id
                    handle = DecodeOverlapPrefetchHandle(
                        req_id=req_id,
                        token_count=intent.token_count,
                        lmcache_hit_tokens=intent.lmcache_hit_tokens,
                        vllm_cached_tokens=intent.vllm_cached_tokens,
                        ready_at=intent.ready_at,
                        request_configs=intent.request_configs,
                        token_signature=planned_signature,
                        mock_only=False,
                        direct_final_slot=False,
                        starts=starts,
                        ends=ends,
                        staged_layers=staged_layers,
                        source_layers=source_layers,
                        layer_ready_events=layer_ready_events,
                    )
                    self._handles[req_id] = handle

                logger.info(
                    "Decode-overlap GPU-compute prefetch partially ready: "
                    "req_id=%s, ready_layers=%d/%d, chunks=%d, tokens=%d, "
                    "location=%s",
                    req_id,
                    layer_id + 1,
                    engine.num_layers,
                    len(starts),
                    ends[-1] - starts[0],
                    location,
                )
            elif handle is not None:
                handle.add_layer(layer_id, staged_layer, source_layer, ready_event)

            if is_proactive and layer_id + 1 >= initial_layer_count:
                logger.info(
                    "Decode-overlap GPU-compute prefetch stopped after configured "
                    "proactive layers: req_id=%s, ready_layers=%d/%d, chunks=%d, "
                    "tokens=%d, location=%s",
                    req_id,
                    layer_id + 1,
                    engine.num_layers,
                    len(starts),
                    ends[-1] - starts[0],
                    location,
                )
                return

        logger.info(
            "Decode-overlap GPU-compute prefetch complete: req_id=%s, layers=%d, "
            "chunks=%d, tokens=%d, location=%s",
            req_id,
            len(staged_layers),
            len(starts),
            ends[-1] - starts[0],
            location,
        )

    def _initial_ready_layer_count(self, req_id: str, num_layers: int) -> int:
        if not req_id.startswith("proactive:"):
            return num_layers
        configured = int(
            os.environ.get("LMCACHE_DECODE_OVERLAP_INITIAL_LAYERS", "4")
        )
        return max(1, min(num_layers, configured))

    def _stage_passive_enabled(self) -> bool:
        return os.environ.get(
            "LMCACHE_DECODE_OVERLAP_STAGE_PASSIVE",
            "False",
        ).lower() in {"1", "true", "yes", "on"}

    def _stage_one_layer_to_gpu_buffer(
        self,
        engine: Any,
        starts: list[int],
        ends: list[int],
        source_layer: list[MemoryObj],
    ) -> tuple[list[StagedMemoryObj], torch.cuda.Event]:
        self._wait_for_decode_window()
        gpu_connector = engine.gpu_connector
        num_all_tokens = ends[-1] - starts[0]
        buf_offset = starts[0]
        buffer_shape = gpu_connector.get_shape(num_all_tokens)
        device = torch.device(gpu_connector.device)
        dtype = gpu_connector.dtype

        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                staged_tensor = torch.empty(
                    buffer_shape,
                    dtype=dtype,
                    device=device,
                )
                staged_tensor.zero_()

                old_positions = torch.zeros(
                    (num_all_tokens,),
                    dtype=torch.int64,
                    device=device,
                )

                for start, end, memory_obj in zip(
                    starts, ends, source_layer, strict=False
                ):
                    assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                    dst_start = start - buf_offset
                    dst_end = end - buf_offset
                    staged_tensor[:, dst_start:dst_end].copy_(
                        memory_obj.tensor,
                        non_blocking=True,
                    )
                    if memory_obj.metadata.cached_positions is not None:
                        old_positions[dst_start:dst_end].copy_(
                            memory_obj.metadata.cached_positions.to(
                                device=device,
                                dtype=torch.int64,
                                non_blocking=True,
                            ),
                            non_blocking=True,
                        )

                metadata = MemoryObjMetadata(
                    shape=staged_tensor.shape,
                    dtype=staged_tensor.dtype,
                    address=staged_tensor.data_ptr(),
                    phy_size=staged_tensor.numel() * staged_tensor.element_size(),
                    ref_count=1,
                    fmt=MemoryFormat.KV_2TD,
                    cached_positions=old_positions,
                )
                staged_layer = [
                    StagedMemoryObj(
                        tensor_value=staged_tensor,
                        metadata=metadata,
                    )
                ]

                ready_event = torch.cuda.Event()
                ready_event.record(stream)

        return staged_layer, ready_event

    def _drain_pending_proactive(self) -> None:
        while True:
            with self._lock:
                if not self._pending_proactive:
                    return
                hint_id, tokens, request_configs = self._pending_proactive.popleft()
            req_id = f"proactive:{hint_id}"
            mask = torch.ones(len(tokens), dtype=torch.bool)
            self.submit_intent(
                req_id=req_id,
                token_count=len(tokens),
                lmcache_hit_tokens=len(tokens),
                vllm_cached_tokens=0,
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
                note="proactive_hint_file",
            )

    def _build_layerwise_plan(
        self,
        engine: Any,
        tokens: list[int],
        mask: torch.Tensor,
        request_configs: Optional[dict[str, Any]],
    ) -> tuple[list[int], list[int], list[list[CacheEngineKey]], Optional[str]]:
        starts: list[int] = []
        ends: list[int] = []
        keys: list[list[CacheEngineKey]] = []
        location = None

        for start, end, key in engine.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys_multi_layer = key.split_layers(engine.num_layers)
            current_location = engine.storage_manager.contains(
                keys_multi_layer[0],
                engine.retrieve_locations,
            )
            if not current_location:
                break
            if location is None:
                location = current_location
            elif location != current_location:
                break
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

        if not keys:
            return [], [], [], location
        keys_layer_major = [list(row) for row in zip(*keys, strict=False)]
        return starts, ends, keys_layer_major, location

    def consume_ready(
        self,
        req_id: str,
        timeout_s: float = 0.0,
    ) -> Optional[DecodeOverlapPrefetchHandle]:
        """Consume a ready prefetch handle for the worker path."""

        deadline = time.time() + max(0.0, timeout_s)
        handle = None
        lookup_req_id = self._resolve_req_id(req_id)
        while True:
            with self._lock:
                intent = self._intents.get(lookup_req_id)
                if intent is None:
                    return None
                if intent.state == "ready":
                    intent.state = "consumed"
                    intent.consumed_at = time.time()
                    handle = self._handles.pop(lookup_req_id, None)
                    if handle is None:
                        handle = DecodeOverlapPrefetchHandle(
                            req_id=req_id,
                            token_count=intent.token_count,
                            lmcache_hit_tokens=intent.lmcache_hit_tokens,
                            vllm_cached_tokens=intent.vllm_cached_tokens,
                            ready_at=intent.ready_at,
                            request_configs=intent.request_configs,
                            token_signature=intent.token_signature,
                        )
                    else:
                        handle.req_id = req_id
                    break

            if time.time() >= deadline:
                return None
            time.sleep(0.001)

        logger.info("Decode-overlap prefetch consumed: req_id=%s", req_id)
        return handle

    def mock_copy_to_vllm_slots(
        self,
        handle: DecodeOverlapPrefetchHandle,
        **kwargs: Any,
    ) -> bool:
        """Mock GPU staging -> vLLM KV slot copy.

        Returns False so callers keep the existing retrieve path.  A real
        implementation should wait on the staging CUDA event, copy from staging
        buffers into the supplied slot_mapping/kvcaches, then return True.
        """

        if handle.mock_only:
            logger.info(
                "Decode-overlap mock copy for req_id=%s. Falling back to normal "
                "LMCache retrieve until GPU staging is implemented.",
                handle.req_id,
            )
        elif handle.direct_final_slot:
            logger.info(
                "Decode-overlap prefetched CPU KV for req_id=%s will be copied "
                "directly into vLLM final KV slots by retrieve_layer.",
                handle.req_id,
            )
        else:
            logger.info(
                "Decode-overlap prefetched GPU KV for req_id=%s will be consumed "
                "by retrieve_layer.",
                handle.req_id,
            )
        return False

    def finish(self, req_id: str) -> None:
        """Drop request-scoped prefetch state after normal request cleanup."""

        with self._lock:
            lookup_req_id = self._resolve_req_id(req_id)
            intent = self._intents.pop(lookup_req_id, None)
            handle = self._handles.pop(lookup_req_id, None)
            self._req_aliases.pop(req_id, None)
            if intent is not None and intent.token_signature is not None:
                self._token_aliases.pop(intent.token_signature, None)
        if handle is not None:
            handle.release_source_refs()

    def cancel(self, req_id: str, reason: str = "") -> None:
        """Cancel and remove a pending or ready prefetch intent."""

        with self._lock:
            lookup_req_id = self._resolve_req_id(req_id)
            intent = self._intents.pop(lookup_req_id, None)
            handle = self._handles.pop(lookup_req_id, None)
            self._req_aliases.pop(req_id, None)
            if intent is not None and intent.token_signature is not None:
                self._token_aliases.pop(intent.token_signature, None)
        if handle is not None:
            handle.release_source_refs()
        if intent is not None:
            logger.info(
                "Decode-overlap prefetch cancelled: req_id=%s, reason=%s",
                req_id,
                reason,
            )

    def _resolve_req_id(self, req_id: str) -> str:
        return self._req_aliases.get(req_id, req_id)

    def _token_signature(self, tokens: list[int]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for token in tokens:
            digest.update(int(token).to_bytes(8, "little", signed=True))
        return digest.hexdigest()

    def _start_hint_watcher_if_configured(self) -> None:
        hint_path = os.environ.get("LMCACHE_DECODE_OVERLAP_PREFETCH_HINT_FILE")
        if not hint_path or self._hint_watcher_started:
            return
        self._hint_watcher_started = True
        interval_s = float(
            os.environ.get("LMCACHE_DECODE_OVERLAP_PREFETCH_HINT_INTERVAL", "0.01")
        )
        thread = threading.Thread(
            target=self._watch_hint_file,
            args=(hint_path, interval_s),
            name=f"{self.name}-decode-overlap-hints",
            daemon=True,
        )
        thread.start()
        logger.info(
            "Decode-overlap proactive hint watcher started: path=%s", hint_path
        )

    def _watch_hint_file(self, hint_path: str, interval_s: float) -> None:
        last_mtime_ns = -1
        while True:
            try:
                stat = os.stat(hint_path)
                if stat.st_mtime_ns != last_mtime_ns:
                    last_mtime_ns = stat.st_mtime_ns
                    self._load_hint_file(hint_path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception(
                    "Decode-overlap failed while reading proactive hint file: %s",
                    hint_path,
                )
            time.sleep(interval_s)

    def _load_hint_file(self, hint_path: str) -> None:
        with open(hint_path) as f:
            payload = json.load(f)
        requests = payload.get("requests", [])
        submitted = 0
        for request in requests:
            hint_id = str(request["id"])
            if hint_id in self._proactive_hint_ids:
                continue
            tokens = [int(token) for token in request["tokens"]]
            if not tokens:
                continue
            self._proactive_hint_ids.add(hint_id)
            self.submit_rag_prefetch(
                hint_id=hint_id,
                tokens=tokens,
                request_configs=request.get("request_configs"),
            )
            submitted += 1
        if submitted:
            logger.info(
                "Decode-overlap submitted %d proactive prefetch hints from %s",
                submitted,
                hint_path,
            )

    def snapshot(self) -> dict[str, str]:
        """Return request id -> state for diagnostics/tests."""

        with self._lock:
            return {req_id: intent.state for req_id, intent in self._intents.items()}

    def _evict_oldest_locked(self) -> None:
        if len(self._intents) < self.max_entries:
            return
        oldest_req_id = min(
            self._intents,
            key=lambda req_id: self._intents[req_id].created_at,
        )
        self._intents.pop(oldest_req_id, None)


class DecodeOverlapPrefetchRegistry:
    """Process-local coordinator registry.

    This is enough for the current single-process mock path.  Multi-process
    deployments should replace it with an RPC-backed prefetch service.
    """

    _coordinators: dict[str, DecodeOverlapPrefetchCoordinator] = {}
    _lock = Lock()

    @classmethod
    def get_or_create(cls, name: str) -> DecodeOverlapPrefetchCoordinator:
        with cls._lock:
            if name not in cls._coordinators:
                cls._coordinators[name] = DecodeOverlapPrefetchCoordinator(name)
            return cls._coordinators[name]
