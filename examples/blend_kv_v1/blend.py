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

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    enable_sparse: bool = False,
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
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt}, sampling_params=sampling_params
    )
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


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

    return parser.parse_args()


def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = args.model

    setup_environment_variables(
        args.use_disk, args.blend_special_str, args.enable_sparse
    )

    tokenizer = AutoTokenizer.from_pretrained(model)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        warmup_prompt = tokenizer.encode("Nice to meet you" * 500)[1:]
        blend_special_str = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]
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
