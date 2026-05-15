# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict, dataclass
import argparse
import contextlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Optional

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


@dataclass
class RequestMetric:
    label: str
    prompt_tokens: int
    total_latency_s: float
    ttft_s: float
    generated_text: str


@dataclass
class BenchmarkSummary:
    name: str
    request_count: int
    total_latency_s: float
    wall_latency_s: float
    avg_latency_s: float
    avg_ttft_s: float
    first_latency_s: float
    reordered_latency_s: float


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
    enable_decode_overlap_prefetch: bool = False,
    proactive_prefetch_hint_file: Optional[str] = None,
    decode_overlap_initial_layers: Optional[int] = None,
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    extra_config = {}
    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        extra_config["enable_sparse"] = True
    if enable_decode_overlap_prefetch:
        extra_config["enable_decode_overlap_prefetch"] = True
        os.environ["LMCACHE_DECODE_OVERLAP_REQUIRE_DECODE_WINDOW"] = "True"
    else:
        os.environ.pop("LMCACHE_DECODE_OVERLAP_REQUIRE_DECODE_WINDOW", None)
    if proactive_prefetch_hint_file is not None:
        os.environ["LMCACHE_DECODE_OVERLAP_PREFETCH_HINT_FILE"] = (
            proactive_prefetch_hint_file
        )
    else:
        os.environ.pop("LMCACHE_DECODE_OVERLAP_PREFETCH_HINT_FILE", None)
    if decode_overlap_initial_layers is not None:
        os.environ["LMCACHE_DECODE_OVERLAP_INITIAL_LAYERS"] = str(
            decode_overlap_initial_layers
        )
    else:
        os.environ.pop("LMCACHE_DECODE_OVERLAP_INITIAL_LAYERS", None)
    if extra_config:
        os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)
    else:
        os.environ.pop("LMCACHE_EXTRA_CONFIG", None)

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=32648,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    metric = run_request(llm, prompt, sampling_params, req_str)
    print("-" * 50)
    print(f"Generated text: {metric.generated_text!r}")
    print(f"Generation took {metric.total_latency_s:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def run_request(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    label: str,
) -> RequestMetric:
    start = time.perf_counter()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    total_latency_s = time.perf_counter() - start
    generated_text = outputs[0].outputs[0].text if outputs else ""
    ttft_s = extract_ttft(outputs[0], total_latency_s) if outputs else total_latency_s
    return RequestMetric(
        label=label,
        prompt_tokens=len(prompt),
        total_latency_s=total_latency_s,
        ttft_s=ttft_s,
        generated_text=generated_text,
    )


def run_batch_requests(
    llm: LLM,
    prompts: list[list[int]],
    labels: list[str],
    sampling_params: SamplingParams,
) -> tuple[float, list[RequestMetric]]:
    start = time.perf_counter()
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": prompt} for prompt in prompts],
        sampling_params=sampling_params,
    )
    batch_latency_s = time.perf_counter() - start

    metrics = []
    for label, prompt, output in zip(labels, prompts, outputs, strict=True):
        generated_text = output.outputs[0].text if output.outputs else ""
        metrics.append(
            RequestMetric(
                label=label,
                prompt_tokens=len(prompt),
                total_latency_s=extract_request_latency(output, batch_latency_s),
                ttft_s=extract_ttft(output, batch_latency_s),
                generated_text=generated_text,
            )
        )
    return batch_latency_s, metrics


def run_sequential_requests(
    llm: LLM,
    prompts: list[list[int]],
    labels: list[str],
    sampling_params: SamplingParams,
) -> tuple[float, list[RequestMetric]]:
    start = time.perf_counter()
    metrics = [
        run_request(llm, prompt, sampling_params, label)
        for label, prompt in zip(labels, prompts, strict=True)
    ]
    return time.perf_counter() - start, metrics


def extract_request_latency(output, fallback_s: float) -> float:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return fallback_s

    arrival_time = getattr(metrics, "arrival_time", None)
    finished_time = getattr(metrics, "finished_time", None)
    if arrival_time is None or finished_time is None:
        return fallback_s
    return max(0.0, finished_time - arrival_time)


def extract_ttft(output, fallback_s: float) -> float:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return fallback_s

    arrival_time = getattr(metrics, "arrival_time", None)
    first_token_time = getattr(metrics, "first_token_time", None)
    if arrival_time is None or first_token_time is None:
        return fallback_s
    return max(0.0, first_token_time - arrival_time)


def measure_ttft(
    llm: LLM,
    prompt: list[int],
    label: str,
) -> float:
    # vLLM offline LLM.generate is not a streaming API, so for this example
    # benchmark we measure TTFT as end-to-end latency for a 1-token generation.
    ttft_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    return run_request(llm, prompt, ttft_params, f"{label} ttft").total_latency_s


def encode_without_bos(tokenizer: AutoTokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text)
    if token_ids and token_ids[0] == tokenizer.bos_token_id:
        return token_ids[1:]
    return token_ids


def build_cacheblend_prompt(
    tokenizer: AutoTokenizer,
    example: dict,
    blend_special_tokens: list[int],
    chunk_order: list[int],
) -> list[int]:
    chunk_prompts = [example[str(i)] for i in range(example["chunk_num"])]
    sys_prompt = encode_without_bos(
        tokenizer,
        "You are a very helpful assistant. "
        "Answer the question based on the given passages.\n\n",
    )
    query_prompt = encode_without_bos(tokenizer, example["query"])

    prompt = sys_prompt
    for chunk_idx in chunk_order:
        prompt += blend_special_tokens
        prompt += encode_without_bos(tokenizer, chunk_prompts[chunk_idx])
    prompt += blend_special_tokens
    prompt += query_prompt
    return prompt


def load_cacheblend_examples(input_dir: str, num_samples: int) -> list[tuple[int, dict]]:
    examples = []
    for sample_idx in range(1, num_samples + 1):
        input_path = Path(input_dir) / f"{sample_idx}.json"
        with open(input_path) as f:
            examples.append((sample_idx, json.load(f)))
    return examples


def build_rag_request_plan(
    tokenizer: AutoTokenizer,
    input_dir: str,
    num_samples: int,
    blend_special_tokens: list[int],
) -> list[tuple[str, str, list[int]]]:
    plan = []
    for sample_idx, example in load_cacheblend_examples(input_dir, num_samples):
        chunk_num = example["chunk_num"]
        original_order = list(range(chunk_num))
        reuse_order = original_order[1:] + original_order[:1]

        first_prompt = build_cacheblend_prompt(
            tokenizer, example, blend_special_tokens, original_order
        )
        second_prompt = build_cacheblend_prompt(
            tokenizer, example, blend_special_tokens, reuse_order
        )

        plan.append((f"sample {sample_idx} first", "first", first_prompt))
        plan.append((f"sample {sample_idx} reordered", "reordered", second_prompt))
    return plan


def summarize_benchmark(
    name: str,
    metrics: list[RequestMetric],
    wall_latency_s: Optional[float] = None,
) -> BenchmarkSummary:
    total_latency_s = sum(metric.total_latency_s for metric in metrics)
    wall_latency_s = total_latency_s if wall_latency_s is None else wall_latency_s
    first_latency_s = sum(
        metric.total_latency_s for metric in metrics if " first" in metric.label
    )
    reordered_latency_s = sum(
        metric.total_latency_s for metric in metrics if " reordered" in metric.label
    )
    avg_latency_s = total_latency_s / len(metrics) if metrics else 0.0
    avg_ttft_s = sum(metric.ttft_s for metric in metrics) / len(metrics) if metrics else 0.0
    return BenchmarkSummary(
        name=name,
        request_count=len(metrics),
        total_latency_s=total_latency_s,
        wall_latency_s=wall_latency_s,
        avg_latency_s=avg_latency_s,
        avg_ttft_s=avg_ttft_s,
        first_latency_s=first_latency_s,
        reordered_latency_s=reordered_latency_s,
    )


def print_metric_table(title: str, metrics: list[RequestMetric]):
    print("\n" + title)
    print("-" * len(title))
    print(f"{'request':<28} {'prompt_tok':>10} {'latency_s':>10} {'ttft_s':>10}")
    for metric in metrics:
        print(
            f"{metric.label:<28} "
            f"{metric.prompt_tokens:>10} "
            f"{metric.total_latency_s:>10.4f} "
            f"{metric.ttft_s:>10.4f}"
        )


def print_summary_table(summaries: list[BenchmarkSummary]):
    print("\nBenchmark summary")
    print("-----------------")
    print(
        f"{'mode':<18} {'reqs':>5} {'wall_s':>10} {'sum_s':>10} "
        f"{'avg_s':>10} {'avg_ttft_s':>12} {'first_s':>10} {'reordered_s':>12}"
    )
    for summary in summaries:
        print(
            f"{summary.name:<18} "
            f"{summary.request_count:>5} "
            f"{summary.wall_latency_s:>10.4f} "
            f"{summary.total_latency_s:>10.4f} "
            f"{summary.avg_latency_s:>10.4f} "
            f"{summary.avg_ttft_s:>12.4f} "
            f"{summary.first_latency_s:>10.4f} "
            f"{summary.reordered_latency_s:>12.4f}"
        )

    if len(summaries) == 2 and summaries[1].wall_latency_s > 0:
        baseline, optimized = summaries
        total_speedup = baseline.wall_latency_s / optimized.wall_latency_s
        ttft_speedup = (
            baseline.avg_ttft_s / optimized.avg_ttft_s
            if optimized.avg_ttft_s > 0
            else 0.0
        )
        latency_reduction = (
            (baseline.wall_latency_s - optimized.wall_latency_s)
            / baseline.wall_latency_s
            * 100
            if baseline.wall_latency_s > 0
            else 0.0
        )
        ttft_reduction = (
            (baseline.avg_ttft_s - optimized.avg_ttft_s)
            / baseline.avg_ttft_s
            * 100
            if baseline.avg_ttft_s > 0
            else 0.0
        )
        print("\nComparison")
        print("----------")
        print(f"Wall-time speedup: {total_speedup:.3f}x")
        print(f"Wall-time reduction: {latency_reduction:.2f}%")
        print(f"Average TTFT speedup: {ttft_speedup:.3f}x")
        print(f"Average TTFT reduction: {ttft_reduction:.2f}%")


def run_rag_benchmark(
    *,
    name: str,
    args,
    tokenizer: AutoTokenizer,
    request_plan: list[tuple[str, str, list[int]]],
    enable_decode_overlap_prefetch: bool,
    measure_first_token_latency: bool,
) -> tuple[BenchmarkSummary, list[RequestMetric]]:
    setup_environment_variables(
        args.use_disk,
        args.blend_special_str,
        args.enable_sparse,
        enable_decode_overlap_prefetch,
        decode_overlap_initial_layers=args.decode_overlap_initial_layers,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    metrics = []
    with build_llm_with_lmcache("LMCacheConnectorV1", args.model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        print_output(llm, warmup_prompt, sampling_params, f"{name} warmup")

        benchmark_start = time.perf_counter()
        for label, _, prompt in request_plan:
            metric = run_request(llm, prompt, sampling_params, label)
            if measure_first_token_latency and args.max_tokens != 1:
                metric.ttft_s = measure_ttft(llm, prompt, label)
            metrics.append(metric)
            print(
                f"[{name}] {label}: latency={metric.total_latency_s:.4f}s, "
                f"ttft={metric.ttft_s:.4f}s, prompt_tokens={metric.prompt_tokens}, "
                f"output={metric.generated_text!r}"
            )
            if args.request_sleep > 0:
                time.sleep(args.request_sleep)
        wall_total_s = time.perf_counter() - benchmark_start

    summary = summarize_benchmark(name, metrics, wall_latency_s=wall_total_s)
    # Keep both numbers visible: sum of per-request latency is stable for
    # sequential runs, while wall total includes optional sleeps.
    print(f"[{name}] benchmark wall time: {wall_total_s:.4f}s")
    return summary, metrics


def run_batched_rag_benchmark(
    *,
    name: str,
    args,
    tokenizer: AutoTokenizer,
    request_plan: list[tuple[str, str, list[int]]],
    enable_decode_overlap_prefetch: bool,
) -> tuple[BenchmarkSummary, list[RequestMetric]]:
    proactive_hint_file = None
    if enable_decode_overlap_prefetch and args.proactive_prefetch_next_batch:
        proactive_hint_file = (
            f"/tmp/lmcache_decode_overlap_hints_{os.getpid()}_{name}.json"
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(proactive_hint_file)

    setup_environment_variables(
        args.use_disk,
        args.blend_special_str,
        args.enable_sparse,
        enable_decode_overlap_prefetch,
        proactive_hint_file,
        decode_overlap_initial_layers=args.decode_overlap_initial_layers,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    first_requests = [
        (label, prompt) for label, kind, prompt in request_plan if kind == "first"
    ]
    reordered_requests = [
        (label, prompt) for label, kind, prompt in request_plan if kind == "reordered"
    ]

    metrics = []
    with build_llm_with_lmcache("LMCacheConnectorV1", args.model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        print_output(llm, warmup_prompt, sampling_params, f"{name} warmup")

        benchmark_start = time.perf_counter()

        first_batch_s, first_metrics = run_batch_requests(
            llm,
            [prompt for _, prompt in first_requests],
            [label for label, _ in first_requests],
            sampling_params,
        )
        metrics.extend(first_metrics)
        print(f"[{name}] first batch wall time: {first_batch_s:.4f}s")
        for metric in first_metrics:
            print(
                f"[{name}] {metric.label}: latency={metric.total_latency_s:.4f}s, "
                f"ttft={metric.ttft_s:.4f}s, prompt_tokens={metric.prompt_tokens}, "
                f"output={metric.generated_text!r}"
            )

        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

        if proactive_hint_file is not None:
            write_proactive_prefetch_hints(
                proactive_hint_file,
                reordered_requests,
            )
            print(
                f"[{name}] proactive prefetch hints written: "
                f"{len(reordered_requests)} requests -> {proactive_hint_file}"
            )
            if args.proactive_prefetch_lead_time > 0:
                time.sleep(args.proactive_prefetch_lead_time)

        reordered_batch_s, reordered_metrics = run_batch_requests(
            llm,
            [prompt for _, prompt in reordered_requests],
            [label for label, _ in reordered_requests],
            sampling_params,
        )
        metrics.extend(reordered_metrics)
        print(f"[{name}] reordered batch wall time: {reordered_batch_s:.4f}s")
        for metric in reordered_metrics:
            print(
                f"[{name}] {metric.label}: latency={metric.total_latency_s:.4f}s, "
                f"ttft={metric.ttft_s:.4f}s, prompt_tokens={metric.prompt_tokens}, "
                f"output={metric.generated_text!r}"
            )

        wall_total_s = time.perf_counter() - benchmark_start

    summary = summarize_benchmark(name, metrics, wall_latency_s=wall_total_s)
    print(f"[{name}] benchmark wall time: {wall_total_s:.4f}s")
    return summary, metrics


def run_rag_decode_overlap_benchmark(
    *,
    name: str,
    args,
    tokenizer: AutoTokenizer,
    request_plan: list[tuple[str, str, list[int]]],
    enable_decode_overlap_prefetch: bool,
) -> tuple[BenchmarkSummary, list[RequestMetric]]:
    """Run a closer-to-RAG decode-overlap experiment.

    Stages:
      0. Build the CPU LMCache by running the source RAG prompts.
      1. Submit a current batch to vLLM and keep it decoding.
      2. A RAG-side thread submits next-batch prefetch hints while stage 1 runs.
      3. Submit the next batch and consume prefetched GPU KV.

    This intentionally keeps the RAG prefetch trigger outside vLLM. In a real
    service this would be owned by the retriever/request scheduler that already
    knows next-batch chunk ids.
    """

    proactive_hint_file = None
    if enable_decode_overlap_prefetch:
        proactive_hint_file = (
            f"/tmp/lmcache_rag_overlap_hints_{os.getpid()}_{name}.json"
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(proactive_hint_file)

    setup_environment_variables(
        args.use_disk,
        args.blend_special_str,
        args.enable_sparse,
        enable_decode_overlap_prefetch,
        proactive_hint_file,
        decode_overlap_initial_layers=args.decode_overlap_initial_layers,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )
    cache_build_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=args.rag_cache_build_max_tokens,
    )

    first_requests = [
        (label, prompt) for label, kind, prompt in request_plan if kind == "first"
    ]
    reordered_requests = [
        (label, prompt) for label, kind, prompt in request_plan if kind == "reordered"
    ]

    metrics = []
    with build_llm_with_lmcache("LMCacheConnectorV1", args.model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        print_output(llm, warmup_prompt, sampling_params, f"{name} warmup")

        print(f"[{name}] Stage 0 cache build: {len(first_requests)} requests")
        cache_build_s, _ = run_batch_requests(
            llm,
            [prompt for _, prompt in first_requests],
            [f"cache build {label}" for label, _ in first_requests],
            cache_build_params,
        )
        print(f"[{name}] Stage 0 cache build wall time: {cache_build_s:.4f}s")

        benchmark_start = time.perf_counter()

        prefetch_thread = None
        if proactive_hint_file is not None:
            prefetch_thread = start_rag_prefetch_thread(
                hint_file=proactive_hint_file,
                requests=reordered_requests,
                delay_s=args.rag_prefetch_start_delay,
                name=name,
            )

        print(
            f"[{name}] Stage 1 current batch serving: "
            f"{len(first_requests)} requests"
        )
        first_batch_s, first_metrics = run_batch_requests(
            llm,
            [prompt for _, prompt in first_requests],
            [label for label, _ in first_requests],
            sampling_params,
        )
        metrics.extend(first_metrics)
        print(f"[{name}] Stage 1 current batch wall time: {first_batch_s:.4f}s")
        for metric in first_metrics:
            print(
                f"[{name}] {metric.label}: latency={metric.total_latency_s:.4f}s, "
                f"ttft={metric.ttft_s:.4f}s, prompt_tokens={metric.prompt_tokens}, "
                f"output={metric.generated_text!r}"
            )

        if prefetch_thread is not None:
            prefetch_thread.join(timeout=args.rag_prefetch_join_timeout)
            if prefetch_thread.is_alive():
                print(
                    f"[{name}] RAG prefetch thread is still running after "
                    f"{args.rag_prefetch_join_timeout:.2f}s; continuing."
                )

        if args.rag_prefetch_after_current_wait > 0:
            time.sleep(args.rag_prefetch_after_current_wait)

        print(
            f"[{name}] Stage 3 next batch serving: "
            f"{len(reordered_requests)} requests"
        )
        reordered_batch_s, reordered_metrics = run_batch_requests(
            llm,
            [prompt for _, prompt in reordered_requests],
            [label for label, _ in reordered_requests],
            sampling_params,
        )
        metrics.extend(reordered_metrics)
        print(f"[{name}] Stage 3 next batch wall time: {reordered_batch_s:.4f}s")
        for metric in reordered_metrics:
            print(
                f"[{name}] {metric.label}: latency={metric.total_latency_s:.4f}s, "
                f"ttft={metric.ttft_s:.4f}s, prompt_tokens={metric.prompt_tokens}, "
                f"output={metric.generated_text!r}"
            )

        wall_total_s = time.perf_counter() - benchmark_start

    summary = summarize_benchmark(name, metrics, wall_latency_s=wall_total_s)
    print(f"[{name}] benchmark wall time: {wall_total_s:.4f}s")
    return summary, metrics


def run_rag_continuous_benchmark(
    *,
    name: str,
    args,
    tokenizer: AutoTokenizer,
    request_plan: list[tuple[str, str, list[int]]],
    enable_decode_overlap_prefetch: bool,
) -> tuple[BenchmarkSummary, list[RequestMetric]]:
    """Run a continuous-serving RAG overlap experiment.

    Stage 0 builds the CPU LMCache for all prompts used in the benchmark.  The
    serving phase then submits a stream of batches.  Batch 0 is the cold-start
    batch and cannot consume proactive KV.  While batch i is running, the RAG
    side submits proactive hints for batch i+1, so later batches model the
    steady-state serving path.
    """

    proactive_hint_file = None
    if enable_decode_overlap_prefetch:
        proactive_hint_file = (
            f"/tmp/lmcache_rag_continuous_hints_{os.getpid()}_{name}.json"
        )
        with contextlib.suppress(FileNotFoundError):
            os.remove(proactive_hint_file)

    setup_environment_variables(
        args.use_disk,
        args.blend_special_str,
        args.enable_sparse,
        enable_decode_overlap_prefetch,
        proactive_hint_file,
        decode_overlap_initial_layers=args.decode_overlap_initial_layers,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )
    cache_build_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=args.rag_cache_build_max_tokens,
    )

    serving_batches = build_continuous_serving_batches(
        request_plan,
        args.continuous_batches,
    )
    cache_build_requests = unique_prompt_requests(
        [
            (label, prompt)
            for batch in serving_batches
            for label, prompt in batch
        ]
    )

    metrics = []
    with build_llm_with_lmcache("LMCacheConnectorV1", args.model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        print_output(llm, warmup_prompt, sampling_params, f"{name} warmup")

        print(
            f"[{name}] Stage 0 continuous cache build: "
            f"{len(cache_build_requests)} unique requests"
        )
        cache_build_s, _ = run_sequential_requests(
            llm,
            [prompt for _, prompt in cache_build_requests],
            [f"cache build {label}" for label, _ in cache_build_requests],
            cache_build_params,
        )
        print(f"[{name}] Stage 0 cache build wall time: {cache_build_s:.4f}s")

        benchmark_start = time.perf_counter()
        prefetch_thread = None

        for batch_idx, batch in enumerate(serving_batches):
            if proactive_hint_file is not None and batch_idx + 1 < len(
                serving_batches
            ):
                prefetch_thread = start_rag_prefetch_thread(
                    hint_file=proactive_hint_file,
                    requests=serving_batches[batch_idx + 1],
                    delay_s=args.rag_prefetch_start_delay,
                    name=f"{name} batch {batch_idx}",
                )

            print(
                f"[{name}] Serving batch {batch_idx}: {len(batch)} requests"
            )
            batch_s, batch_metrics = run_batch_requests(
                llm,
                [prompt for _, prompt in batch],
                [label for label, _ in batch],
                sampling_params,
            )
            metrics.extend(batch_metrics)
            print(f"[{name}] Serving batch {batch_idx} wall time: {batch_s:.4f}s")
            for metric in batch_metrics:
                print(
                    f"[{name}] {metric.label}: "
                    f"latency={metric.total_latency_s:.4f}s, "
                    f"ttft={metric.ttft_s:.4f}s, "
                    f"prompt_tokens={metric.prompt_tokens}, "
                    f"output={metric.generated_text!r}"
                )

            if prefetch_thread is not None:
                prefetch_thread.join(timeout=args.rag_prefetch_join_timeout)
                if prefetch_thread.is_alive():
                    print(
                        f"[{name}] RAG prefetch thread for batch {batch_idx + 1} "
                        f"is still running after "
                        f"{args.rag_prefetch_join_timeout:.2f}s; continuing."
                    )
                prefetch_thread = None

        wall_total_s = time.perf_counter() - benchmark_start

    summary = summarize_benchmark(name, metrics, wall_latency_s=wall_total_s)
    print(f"[{name}] continuous benchmark wall time: {wall_total_s:.4f}s")
    return summary, metrics


def build_continuous_serving_batches(
    request_plan: list[tuple[str, str, list[int]]],
    num_batches: int,
) -> list[list[tuple[str, list[int]]]]:
    first_requests = [
        (label, prompt) for label, kind, prompt in request_plan if kind == "first"
    ]
    reordered_requests = [
        (label, prompt) for label, kind, prompt in request_plan if kind == "reordered"
    ]
    base_batches = [first_requests, reordered_requests]
    batches = []
    for batch_idx in range(num_batches):
        source_batch = base_batches[batch_idx % len(base_batches)]
        batches.append(
            [
                (f"batch {batch_idx} {label}", prompt)
                for label, prompt in source_batch
            ]
        )
    return batches


def unique_prompt_requests(
    requests: list[tuple[str, list[int]]],
) -> list[tuple[str, list[int]]]:
    seen = set()
    unique = []
    for label, prompt in requests:
        key = tuple(prompt)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, prompt))
    return unique


def start_rag_prefetch_thread(
    *,
    hint_file: str,
    requests: list[tuple[str, list[int]]],
    delay_s: float,
    name: str,
) -> threading.Thread:
    def run() -> None:
        if delay_s > 0:
            time.sleep(delay_s)
        write_proactive_prefetch_hints(hint_file, requests)
        print(
            f"[{name}] Stage 2 RAG prefetch hints written during current "
            f"batch: {len(requests)} requests -> {hint_file}"
        )

    thread = threading.Thread(
        target=run,
        name=f"{name}-rag-prefetch-thread",
        daemon=True,
    )
    thread.start()
    return thread


def write_proactive_prefetch_hints(
    hint_file: str,
    requests: list[tuple[str, list[int]]],
) -> None:
    payload = {
        "created_at": time.time(),
        "requests": [
            {
                "id": label,
                "label": label,
                "tokens": prompt,
            }
            for label, prompt in requests
        ],
    }
    tmp_file = f"{hint_file}.tmp"
    with open(tmp_file, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_file, hint_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "--enable-sparse",
        action="store_true",
    )

    parser.add_argument(
        "--enable-decode-overlap-prefetch",
        action="store_true",
        help=(
            "Enable decode-overlap KV prefetch. This stages LMCache KV from "
            "CPU to GPU asynchronously when a scheduler-side hit is detected."
        ),
    )

    parser.add_argument(
        "--compare-decode-overlap-prefetch",
        action="store_true",
        help=(
            "Run two RAG benchmark passes: baseline LMCache blend first, then "
            "decode-overlap CPU->GPU prefetch, and print a comparison table."
        ),
    )

    parser.add_argument(
        "--benchmark-mode",
        choices=("batch", "sequential", "rag-overlap", "rag-continuous"),
        default="batch",
        help=(
            "Benchmark request submission mode for comparison runs. 'batch' "
            "submits all first RAG prompts together, then all reordered hit "
            "prompts together, exposing cross-request overlap opportunities "
            "(default: batch). 'rag-overlap' first builds CPU cache, then "
            "submits RAG prefetch hints from a side thread while the current "
            "batch is running. 'rag-continuous' repeats that policy over a "
            "continuous stream of batches."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/sco/code/CacheBlend/inputs",
        help="Directory containing CacheBlend JSON samples.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of CacheBlend numbered samples to run.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Number of tokens to generate for each request.",
    )

    parser.add_argument(
        "--measure-ttft-with-one-token",
        action="store_true",
        help=(
            "Measure TTFT with an extra max_tokens=1 request for each prompt. "
            "Offline vLLM LLM.generate is non-streaming, so this is the most "
            "direct first-token latency proxy in this example."
        ),
    )

    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between benchmark requests (default: 0.0).",
    )

    parser.add_argument(
        "--proactive-prefetch-next-batch",
        action="store_true",
        help=(
            "In batch benchmark mode, write next-batch RAG prompt hints after "
            "the first batch so the LMCache worker can prefetch reordered "
            "request KV before those requests are submitted to vLLM."
        ),
    )

    parser.add_argument(
        "--proactive-prefetch-lead-time",
        type=float,
        default=0.2,
        help=(
            "Seconds to wait after writing proactive prefetch hints before "
            "submitting the reordered batch (default: 0.2)."
        ),
    )

    parser.add_argument(
        "--rag-prefetch-start-delay",
        type=float,
        default=0.05,
        help=(
            "In --benchmark-mode rag-overlap, seconds after current-batch "
            "submission starts before the RAG-side thread writes next-batch "
            "prefetch hints (default: 0.05)."
        ),
    )

    parser.add_argument(
        "--rag-prefetch-join-timeout",
        type=float,
        default=5.0,
        help=(
            "In --benchmark-mode rag-overlap, maximum seconds to wait for the "
            "RAG-side hint writer after current batch completes (default: 5.0)."
        ),
    )

    parser.add_argument(
        "--rag-prefetch-after-current-wait",
        type=float,
        default=0.0,
        help=(
            "In --benchmark-mode rag-overlap, optional seconds to wait after "
            "current batch completes before submitting next batch. Keep this "
            "at 0 to avoid counting artificial lead time (default: 0.0)."
        ),
    )

    parser.add_argument(
        "--rag-cache-build-max-tokens",
        type=int,
        default=1,
        help=(
            "In --benchmark-mode rag-overlap, max tokens used during Stage 0 "
            "CPU cache build requests (default: 1)."
        ),
    )

    parser.add_argument(
        "--decode-overlap-initial-layers",
        type=int,
        default=4,
        help=(
            "Number of early layers to proactively prefetch before marking a "
            "RAG hint ready. Remaining layers continue to prefetch in the "
            "background while the next batch recomputes (default: 4)."
        ),
    )

    parser.add_argument(
        "--continuous-batches",
        type=int,
        default=4,
        help=(
            "In --benchmark-mode rag-continuous, number of serving batches to "
            "run after Stage 0 cache build (default: 4)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    tokenizer = AutoTokenizer.from_pretrained(model)
    blend_special_str = tokenizer.encode(args.blend_special_str)[1:]

    if args.compare_decode_overlap_prefetch:
        request_plan = build_rag_request_plan(
            tokenizer,
            args.input_dir,
            args.num_samples,
            blend_special_str,
        )
        summaries = []
        if args.benchmark_mode == "batch":
            benchmark_fn = run_batched_rag_benchmark
        elif args.benchmark_mode == "rag-overlap":
            benchmark_fn = run_rag_decode_overlap_benchmark
        elif args.benchmark_mode == "rag-continuous":
            benchmark_fn = run_rag_continuous_benchmark
        else:
            benchmark_fn = run_rag_benchmark

        if args.benchmark_mode in {"batch", "rag-overlap", "rag-continuous"} and (
            args.measure_ttft_with_one_token
        ):
            print(
                "Ignoring --measure-ttft-with-one-token in batch/rag-overlap/"
                "rag-continuous mode; using vLLM RequestOutput.metrics when "
                "available."
            )

        common_kwargs = dict(
            args=args,
            tokenizer=tokenizer,
            request_plan=request_plan,
        )

        if args.benchmark_mode in {"batch", "rag-overlap", "rag-continuous"}:
            baseline_summary, baseline_metrics = benchmark_fn(
                name="baseline",
                enable_decode_overlap_prefetch=False,
                **common_kwargs,
            )
        else:
            baseline_summary, baseline_metrics = benchmark_fn(
                name="baseline",
                enable_decode_overlap_prefetch=False,
                measure_first_token_latency=args.measure_ttft_with_one_token,
                **common_kwargs,
            )
        summaries.append(baseline_summary)
        print_metric_table("Baseline per-request metrics", baseline_metrics)

        if args.benchmark_mode in {"batch", "rag-overlap", "rag-continuous"}:
            prefetch_summary, prefetch_metrics = benchmark_fn(
                name="prefetch",
                enable_decode_overlap_prefetch=True,
                **common_kwargs,
            )
        else:
            prefetch_summary, prefetch_metrics = benchmark_fn(
                name="prefetch",
                enable_decode_overlap_prefetch=True,
                measure_first_token_latency=args.measure_ttft_with_one_token,
                **common_kwargs,
            )
        summaries.append(prefetch_summary)
        print_metric_table("Prefetch per-request metrics", prefetch_metrics)
        print_summary_table(summaries)
        return

    setup_environment_variables(
        args.use_disk,
        args.blend_special_str,
        args.enable_sparse,
        args.enable_decode_overlap_prefetch,
        decode_overlap_initial_layers=args.decode_overlap_initial_layers,
    )

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        sampling_params = SamplingParams(
            temperature=0, top_p=0.95, max_tokens=args.max_tokens
        )

        print_output(llm, warmup_prompt, sampling_params, "warmup")

        for sample_idx, example in load_cacheblend_examples(
            args.input_dir, args.num_samples
        ):
            chunk_num = example["chunk_num"]
            original_order = list(range(chunk_num))
            reuse_order = original_order[1:] + original_order[:1]

            first_prompt = build_cacheblend_prompt(
                tokenizer, example, blend_special_str, original_order
            )
            second_prompt = build_cacheblend_prompt(
                tokenizer, example, blend_special_str, reuse_order
            )

            print(f"Running CacheBlend sample {sample_idx} ({chunk_num} chunks)")
            print_output(llm, first_prompt, sampling_params, f"sample {sample_idx} first")

            time.sleep(1)

            print_output(
                llm,
                second_prompt,
                sampling_params,
                f"sample {sample_idx} reordered",
            )


if __name__ == "__main__":
    main()
