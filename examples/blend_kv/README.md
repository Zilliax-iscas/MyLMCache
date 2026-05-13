# KV blending 示例
这是一个最小示例，用来演示 LMCache 的 KV blending 功能。

通过在配置 YAML 中设置 `enable_blending: True` 来启用 KV blending 功能。

在 `blend_kv.py` 中，下面的代码会先计算两个文本 chunk 的 KV cache。
```python
offline_precompute = OfflineKVPreCompute(llm)
for chunk in chunks:
    offline_precompute.precompute_kv(chunk)
```

然后，将这些文本 chunk 拼接在一起，在前面加上 system prompt，在后面加上用户的问题。
```python
user_prompt= [sys_prompt, chunks[0], chunks[1], question]
user_prompt = combine_input_prompt_chunks(user_prompt)
```

最后，这个 prompt 会被发送到 serving engine，KV blending 模块会对这些文本 chunk 的 KV 进行 blending。


## 如何运行
### 离线
```
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False python3 blend_kv.py
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False python3 batched_kv.py
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False VLLM_WORKER_MULTIPROC_METHOD=spawn python3 tp_kv.py
LMCACHE_CONFIG_FILE=example_blending.yaml VLLM_WORKER_MULTIPROC_METHOD=spawn LMCACHE_USE_EXPERIMENTAL=False python3 batched_tp_kv.py
```
### 在线
```
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False CUDA_VISIBLE_DEVICES=0 python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000
python3 online_kv.py 8000
```
```
LMCACHE_CONFIG_FILE=example_blending.yaml LMCACHE_USE_EXPERIMENTAL=False CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --tensor-parallel-size 2
python3 online_kv.py 8000
```
