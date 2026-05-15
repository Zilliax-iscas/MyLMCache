# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import asdict
import argparse
import contextlib
import json
import os
from pathlib import Path
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


MUSIQUE_PREFIX_PROMPT = (
    "You will be asked a question after reading several passages. "
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words..\n"
    "Passages:\n"
)
MUSIQUE_QUERY_PROMPT = (
    "\n\nAnswer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. \n"
    "Question:"
)


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
    profile_blend_transfer: bool = False,
    local_cpu_size_gb: float = 16.0,
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.5"
    os.environ["LMCACHE_BLEND_PROFILE"] = "True" if profile_blend_transfer else "False"
    if profile_blend_transfer:
        profile_path = f"/tmp/lmcache_blend_profile_{os.getpid()}.jsonl"
        if os.path.exists(profile_path):
            os.remove(profile_path)
        os.environ["LMCACHE_BLEND_PROFILE_FILE"] = profile_path
    else:
        os.environ.pop("LMCACHE_BLEND_PROFILE_FILE", None)

    if enable_sparse:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        os.environ["LMCACHE_EXTRA_CONFIG"] = '{"enable_sparse": true}'

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

        # MuSiQue samples are much larger than the original 10 toy inputs.
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(local_cpu_size_gb)


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
    timing_records: list[dict] | None = None,
    profile_reader_state: dict | None = None,
) -> dict:
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    elapsed = time.time() - start
    blend_profile = read_latest_blend_profile(profile_reader_state)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        print(f"Generated tokens: {len(output.outputs[0].token_ids)}")
    print(f"Generation took {elapsed:.2f} seconds, {req_str} request done.")
    record = print_blend_transfer_profile(elapsed, req_str, blend_profile)
    if timing_records is not None:
        timing_records.append(record)
    print("-" * 50)
    return record


def read_latest_blend_profile(profile_reader_state: dict | None) -> dict | None:
    if profile_reader_state is None:
        return None

    profile_path = os.getenv("LMCACHE_BLEND_PROFILE_FILE")
    if not profile_path or not os.path.exists(profile_path):
        return None

    records = []
    with open(profile_path) as profile_file:
        profile_file.seek(profile_reader_state.get("offset", 0))
        for line in profile_file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        profile_reader_state["offset"] = profile_file.tell()

    if not records:
        return None
    return records[-1]


def print_blend_transfer_profile(
    generation_elapsed_s: float,
    req_str: str,
    profile: dict | None,
) -> dict:
    record = {
        "request": req_str,
        "generation_ms": generation_elapsed_s * 1000,
        "cpu_to_gpu_kv_ms": 0.0,
        "cpu_to_gpu_kv_pct": 0.0,
        "gpu_paged_write_ms": 0.0,
        "combined_kv_movement_ms": 0.0,
        "combined_kv_movement_pct": 0.0,
        "blend_total_ms": 0.0,
        "blend_total_pct": 0.0,
        "retrieve_step_wall_ms": 0.0,
        "model_step_wall_ms": 0.0,
        "has_blend_profile": False,
    }

    if not profile:
        print_timing_record(record)
        return record

    generation_ms = generation_elapsed_s * 1000
    cpu_to_gpu_kv_ms = float(profile.get("h2d_total_ms", 0.0))
    gpu_paged_write_ms = float(profile.get("paged_write_total_ms", 0.0))
    combined_kv_movement_ms = float(profile.get("kv_transfer_total_ms", 0.0))
    blend_total_ms = float(profile.get("blend_total_ms", 0.0))
    retrieve_wall_ms = float(profile.get("retrieve_step_wall_ms", 0.0))
    model_wall_ms = float(profile.get("model_step_wall_ms", 0.0))

    if generation_ms <= 0:
        print_timing_record(record)
        return record

    record.update(
        {
            "cpu_to_gpu_kv_ms": cpu_to_gpu_kv_ms,
            "cpu_to_gpu_kv_pct": cpu_to_gpu_kv_ms / generation_ms * 100,
            "gpu_paged_write_ms": gpu_paged_write_ms,
            "combined_kv_movement_ms": combined_kv_movement_ms,
            "combined_kv_movement_pct": combined_kv_movement_ms
            / generation_ms
            * 100,
            "blend_total_ms": blend_total_ms,
            "blend_total_pct": blend_total_ms / generation_ms * 100,
            "retrieve_step_wall_ms": retrieve_wall_ms,
            "model_step_wall_ms": model_wall_ms,
            "has_blend_profile": True,
        }
    )

    print_timing_record(record)
    return record


def print_timing_record(record: dict):
    print("KV Cache Reuse Timing")
    print("-" * 78)
    print(f"Request: {record['request']}")
    print(f"Total inference wall time: {record['generation_ms']:.3f} ms")
    if not record["has_blend_profile"]:
        print("No layerwise KV reuse profile was produced for this request.")
        print("-" * 78)
        return

    print(f"{'Metric':<36} {'Time (ms)':>14} {'Share of total':>18}")
    print("-" * 78)
    print(
        f"{'CPU -> GPU KV transfer':<36} "
        f"{record['cpu_to_gpu_kv_ms']:>14.3f} "
        f"{record['cpu_to_gpu_kv_pct']:>17.2f}%"
    )
    print(
        f"{'GPU buffer -> paged KV write':<36} "
        f"{record['gpu_paged_write_ms']:>14.3f} "
        f"{record['gpu_paged_write_ms'] / record['generation_ms'] * 100:>17.2f}%"
    )
    print(
        f"{'Combined KV movement':<36} "
        f"{record['combined_kv_movement_ms']:>14.3f} "
        f"{record['combined_kv_movement_pct']:>17.2f}%"
    )
    print(
        f"{'Layerwise retrieve wall time':<36} "
        f"{record['retrieve_step_wall_ms']:>14.3f} "
        f"{record['retrieve_step_wall_ms'] / record['generation_ms'] * 100:>17.2f}%"
    )
    print(
        f"{'Layerwise model wall time':<36} "
        f"{record['model_step_wall_ms']:>14.3f} "
        f"{record['model_step_wall_ms'] / record['generation_ms'] * 100:>17.2f}%"
    )
    print(
        f"{'End-to-end blend phase':<36} "
        f"{record['blend_total_ms']:>14.3f} "
        f"{record['blend_total_pct']:>17.2f}%"
    )
    print("-" * 78)


def print_timing_summary(records: list[dict]):
    profiled_records = [record for record in records if record["has_blend_profile"]]
    reordered_records = [
        record
        for record in profiled_records
        if record["request"].endswith("reordered")
    ]

    print("=" * 96)
    print("Aggregate KV Cache Reuse Timing")
    print("=" * 96)
    print_aggregate_records("all profiled requests", profiled_records)
    print_aggregate_records("reordered requests", reordered_records)
    print("=" * 96)


def print_aggregate_records(label: str, records: list[dict]):
    if not records:
        print(f"{label}: no profiled KV retrieve/blend requests")
        return

    total_generation_ms = sum(record["generation_ms"] for record in records)
    total_cpu_to_gpu_kv_ms = sum(record["cpu_to_gpu_kv_ms"] for record in records)
    total_gpu_paged_write_ms = sum(record["gpu_paged_write_ms"] for record in records)
    total_combined_kv_movement_ms = sum(
        record["combined_kv_movement_ms"] for record in records
    )
    total_blend_ms = sum(record["blend_total_ms"] for record in records)

    print(f"{label}:")
    print(f"  requests: {len(records)}")
    print(f"  total_inference_time: {total_generation_ms:.3f} ms")
    print(f"{'Metric':<36} {'Total (ms)':>14} {'Share of total':>18}")
    print("-" * 78)
    print(
        f"{'CPU -> GPU KV transfer':<36} "
        f"{total_cpu_to_gpu_kv_ms:>14.3f} "
        f"{total_cpu_to_gpu_kv_ms / total_generation_ms * 100:>17.2f}%"
    )
    print(
        f"{'GPU buffer -> paged KV write':<36} "
        f"{total_gpu_paged_write_ms:>14.3f} "
        f"{total_gpu_paged_write_ms / total_generation_ms * 100:>17.2f}%"
    )
    print(
        f"{'Combined KV movement':<36} "
        f"{total_combined_kv_movement_ms:>14.3f} "
        f"{total_combined_kv_movement_ms / total_generation_ms * 100:>17.2f}%"
    )
    print(
        f"{'End-to-end blend phase':<36} "
        f"{total_blend_ms:>14.3f} "
        f"{total_blend_ms / total_generation_ms * 100:>17.2f}%"
    )
    print("-" * 78)


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


def normalize_question(question: str) -> str:
    if not question.endswith("?"):
        question += "?"
    return question[0].lower() + question[1:]


def build_musique_prompt(
    tokenizer: AutoTokenizer,
    example: dict,
    blend_special_tokens: list[int],
    passage_order: list[int],
) -> list[int]:
    prompt = encode_without_bos(tokenizer, MUSIQUE_PREFIX_PROMPT)

    passage_prompts = [
        f"{ctx.get('title', '')}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]
    ]
    for passage_idx in passage_order:
        prompt += blend_special_tokens
        prompt += encode_without_bos(tokenizer, passage_prompts[passage_idx])

    question = normalize_question(example["question"])
    prompt += blend_special_tokens
    prompt += encode_without_bos(
        tokenizer,
        f"{MUSIQUE_QUERY_PROMPT}{question}\nAnswer:",
    )
    return prompt


def load_cacheblend_examples(input_dir: str, num_samples: int) -> list[tuple[int, dict]]:
    examples = []
    for sample_idx in range(1, num_samples + 1):
        input_path = Path(input_dir) / f"{sample_idx}.json"
        with open(input_path) as f:
            examples.append((sample_idx, json.load(f)))
    return examples


def load_musique_examples(input_path: str, num_samples: int) -> list[tuple[int, dict]]:
    with open(input_path) as f:
        dataset = json.load(f)
    if num_samples > 0:
        dataset = dataset[:num_samples]
    return [(sample_idx, example) for sample_idx, example in enumerate(dataset, start=1)]


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
        "--input-dir",
        type=str,
        default="/home/sco/code/CacheBlend/inputs",
        help="Directory containing CacheBlend JSON samples.",
    )
    parser.add_argument(
        "--dataset",
        choices=["cacheblend", "musique"],
        default="cacheblend",
        help="Dataset format to run.",
    )
    parser.add_argument(
        "--musique-input",
        type=str,
        default="/home/sco/code/CacheBlend/inputs/musique_s.json",
        help="Path to the MuSiQue JSON array.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to run. Use <= 0 for all MuSiQue examples.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Number of tokens to generate for each request.",
    )
    parser.add_argument(
        "--profile-blend-transfer",
        action="store_true",
        help="Print CacheBlend KV transfer timing and generation percentage.",
    )
    parser.add_argument(
        "--local-cpu-size-gb",
        type=float,
        default=16.0,
        help="Maximum LMCache LocalCPU cache size in GiB.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(
        args.use_disk,
        args.blend_special_str,
        args.enable_sparse,
        args.profile_blend_transfer,
        args.local_cpu_size_gb,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]
        sampling_params = SamplingParams(
            temperature=0, top_p=0.95, max_tokens=args.max_tokens
        )
        timing_records = []
        profile_reader_state = {"offset": 0}

        print_output(
            llm,
            warmup_prompt,
            sampling_params,
            "warmup",
            timing_records,
            profile_reader_state,
        )

        if args.dataset == "cacheblend":
            examples = load_cacheblend_examples(args.input_dir, args.num_samples)
            for sample_idx, example in examples:
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
                print_output(
                    llm,
                    first_prompt,
                    sampling_params,
                    f"sample {sample_idx} first",
                    timing_records,
                    profile_reader_state,
                )

                time.sleep(1)

                print_output(
                    llm,
                    second_prompt,
                    sampling_params,
                    f"sample {sample_idx} reordered",
                    timing_records,
                    profile_reader_state,
                )
        else:
            examples = load_musique_examples(args.musique_input, args.num_samples)
            for sample_idx, example in examples:
                passage_num = len(example["ctxs"])
                original_order = list(range(passage_num))
                reuse_order = original_order[1:] + original_order[:1]

                first_prompt = build_musique_prompt(
                    tokenizer, example, blend_special_str, original_order
                )
                second_prompt = build_musique_prompt(
                    tokenizer, example, blend_special_str, reuse_order
                )

                print(
                    f"Running MuSiQue sample {sample_idx} "
                    f"({passage_num} passages, answers={example.get('answers', [])})"
                )
                print_output(
                    llm,
                    first_prompt,
                    sampling_params,
                    f"musique {sample_idx} first",
                    timing_records,
                    profile_reader_state,
                )

                time.sleep(1)

                print_output(
                    llm,
                    second_prompt,
                    sampling_params,
                    f"musique {sample_idx} reordered",
                    timing_records,
                    profile_reader_state,
                )

        print_timing_summary(timing_records)


if __name__ == "__main__":
    main()
