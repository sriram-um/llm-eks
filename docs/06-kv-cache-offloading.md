# 06 — KV cache offloading with LMCache: L1 (CPU RAM) and L2 (Valkey)

Chapter 05 ended with `Prefix cache hits % = 0` and a 27-second TTFT under load. This chapter
attacks the root cause: **recomputing KV tensors for prompt text the GPU has already seen.**

Results from the run:

| Configuration | Cold TTFT | Warm TTFT | Speedup |
|---|---|---|---|
| L1 — CPU RAM only | 2.753 s | **0.132 s** | **20.9×** |
| L1 + L2 — Valkey, same pod | 2.907 s | **0.202 s** | **14.4×** |
| L1 + L2 — **freshly recreated pod** | 4.687 s | **0.400 s** | **11.7×** |

That third row is the one that matters architecturally: a brand-new pod with an empty local
cache still served a warm request in 400 ms, because the KV tensors were waiting in Valkey.

> **In plain terms.** When a model reads your prompt, it computes an internal summary of every
> token — the **KV cache** — so that later tokens don't have to re-read the earlier ones from
> scratch. That work is expensive, and it's **thrown away** when the request ends.
>
> Now notice how much text repeats in real applications. A system prompt is identical on every
> single call. A chat conversation resends the entire history each turn. A RAG app prepends the
> same instruction block every time. Every one of those is the GPU redoing arithmetic it already
> did, for an answer it already knows.
>
> So: keep the answer. And when GPU memory fills up, put it somewhere slower rather than throwing
> it out — exactly the way a CPU has L1, L2 and RAM behind its registers:
>
> ```
>   GPU HBM        fastest, smallest   ← vLLM's own prefix cache, one pod, dies with it
>   CPU RAM (L1)   ~11 GB/s            ← same node, ~100× more room
>   Valkey  (L2)   ~0.24 GB/s          ← over the network, SHARED and SURVIVES the pod
>   recompute      slowest of all      ← what you're trying to avoid
> ```
>
> The counter-intuitive result is the point of the chapter: **fetching a cache over the network at
> 0.24 GB/s still beats recomputing it on a GPU.** It sounds wrong — a network is obviously
> slower than a GPU — but you're not comparing network to GPU, you're comparing *copying a
> finished result* to *redoing the whole prefill*. Copying wins by 11×.
>
> Two consequences that matter in production:
>
> - **L2 is shared.** Ten replicas warm each other's cache instead of each learning alone.
> - **L2 survives restarts.** A pod that has just been rescheduled is already warm — which is the
>   third row of the table above, and the reason this is an architecture decision and not a
>   micro-optimisation.
>
> The cost side is measured too, and it's small: writing to the cache took **48 ms** for a
> 4,096-token prompt, of which the network `put` was **0.13 ms**. Almost all of it is moving data
> off the GPU, not the network.
>
> *Background:* [CONCEPTS Part 1](CONCEPTS.md#the-kv-cache-why-memory-not-compute-is-your-capacity-limit)
> — what the KV cache is and why it's ~136 KB per token.

## Why this works

vLLM's GPU prefix cache already reuses KV tensors for identical prompt prefixes — but only
what fits in GPU HBM, and only within one pod's lifetime. Chapter 03's dashboard showed
40.6 GB of the 48 GB card already committed, so there is very little room.

LMCache adds a memory hierarchy underneath it:

```
GPU HBM (vLLM prefix cache)   ~28 GiB   ns          scoped to one pod, evicted constantly
        ↓ miss
L1: CPU RAM (LMCache)         ~100s GB  ~10 GB/s    scoped to one pod, survives requests
        ↓ miss
L2: ElastiCache Valkey        unbounded ~0.24 GB/s  SHARED across pods, survives pod deletion
        ↓ miss
Recompute on the GPU                                 the thing you're avoiding
```

Every layer is slower than the one above and dramatically faster than recomputing. Real
workloads have enormous prefix overlap — a shared system prompt, multi-turn conversations
replaying history, RAG with a stable instruction block, few-shot examples — so the hit rate is
nothing like the 0 % a synthetic benchmark produces.

## Part 1 — L1: offload to CPU RAM

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

envsubst < 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-pod.yaml > /tmp/lmcache-cpu-ram-pod.yaml \
  && mv /tmp/lmcache-cpu-ram-pod.yaml 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-pod.yaml

cat 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-pod.yaml | grep -A 2 "model=s3://"

kubectl apply -f 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-configmap.yaml
kubectl apply -f 400-lmcache/cpu-ram-offloading/lmcache-cpu-ram-pod.yaml
```

```
        --model=s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/ \
        --served-model-name=ministral \
        --load-format=runai_streamer \
configmap/cpu-ram-cm created
pod/lmcache-cpu-ram created
```

Same model, same S3 loading path as Chapter 02 — the only additions are the LMCache KV
connector configuration (mounted from the `cpu-ram-cm` ConfigMap) and vLLM listening on
`:8080` instead of `:8000`. This is a bare `Pod`, not a Deployment: it's an experiment, and
deleting it in Part 2 is the point of the experiment.

```bash
kubectl wait --for=condition=ready pod/lmcache-cpu-ram --timeout=600s \
  && kubectl logs lmcache-cpu-ram --tail=50
```

```
pod/lmcache-cpu-ram condition met
LMCache INFO: Transport connecting to rank 0 with socket path /tmp/engine_…_lmcache_rpc_port_0
LMCache WARNING: Could not load 'builtin' from vLLM. Using builtin hash. This may cause
                 inconsistencies in distributed caching.
LMCache WARNING: Using builtin hash without PYTHONHASHSEED set. For production environments
                 (non-testing scenarios), you MUST set PYTHONHASHSEED to ensure consistent
                 hashing across processes. Example: export PYTHONHASHSEED=0
LMCache INFO: Using hash algorithm: builtin
LMCache INFO: LMCache initialized for role KVConnectorRole.SCHEDULER with version
              0.4.5-ga7934b79, vllm version 0.21.0
INFO [api_server.py:617] Starting vLLM server on http://0.0.0.0:8080
INFO:     Application startup complete.
```

> **Read that `PYTHONHASHSEED` warning.** Cache keys are hashes of token sequences. Python
> randomizes `hash()` per process by default, so two pods — or the same pod after a restart —
> compute *different keys for identical prompts*. Every lookup misses and your shared L2 cache
> silently does nothing. Set `PYTHONHASHSEED=0` in production. See
> [GOTCHAS](GOTCHAS.md#lmcache-pythonhashseed).

### Measure it

```bash
kubectl exec lmcache-cpu-ram -- python /app/query-twice.py
```

```
============================================================
LMCache L1 (CPU RAM) Cache Test
============================================================
Sends two requests with 90% prompt overlap.
Cold request populates the L1 cache; warm request should hit L1
and return faster (lower TTFT).

=== Cold request: LMCache empty, GPU computes all tokens ===
Sending 4,092-token prompt to vLLM (model streaming response):
  Response preview: The **`os`** module in Python provides platform-independent functions and attrib...
[OK] Cold TTFT: 2.753s (L1 cache now populated with prompt's KV tensors)

=== Warm request: same 90% content, LMCache L1 should serve the matching prefix ===
Sending 4,039-token prompt to vLLM (model streaming response):
  Response preview: The `os` module in Python provides platform-independent access to operating syst...
[OK] Warm TTFT: 0.132s (GPU only computed the new 10% tail)

------------------------------------------------------------
TTFT Improvement: 2.622s (20.9x faster)
Why: cold had to compute KV tensors for all tokens in the prompt;
warm served ~90% from LMCache L1 CPU RAM, so the GPU only had to
compute the new ~10% (tail bytes + question) from scratch.
------------------------------------------------------------
```

**2.753 s → 0.132 s.** The test is honest about what it's measuring: two prompts with 90 %
overlap, not identical prompts. The GPU still prefilled the ~10 % tail plus the new question.
The other 90 % came out of CPU RAM.

Scale that to a chat application: turn 20 of a conversation replays the entire history as
prefix. Without caching you re-prefill all of it on every turn, and prefill cost grows with
conversation length. With caching you prefill only the new message.

```bash
kubectl delete pod lmcache-cpu-ram
kubectl delete configmap cpu-ram-cm
```

## Part 2 — L2: share the cache across pods with ElastiCache for Valkey

L1 dies with the pod. In a real deployment with N replicas behind a Service, each pod builds
its own private L1 and a user's second request — load-balanced to a different pod — misses.
L2 fixes that.

```bash
export CACHE_NAME="lmcache-valkey-eks"

aws elasticache describe-serverless-caches \
  --serverless-cache-name "$CACHE_NAME" \
  --region "$AWS_REGION" \
  --query 'ServerlessCaches[0].{Status:Status,Engine:Engine,Endpoint:Endpoint.Address,Port:Endpoint.Port}' \
  --output table

export VALKEY_ENDPOINT=$(aws elasticache describe-serverless-caches \
  --serverless-cache-name "$CACHE_NAME" --region "$AWS_REGION" \
  --query 'ServerlessCaches[0].Endpoint.Address' --output text)

echo "Valkey endpoint: $VALKEY_ENDPOINT"
```

```
--------------------------------------------------------------------------
|                       DescribeServerlessCaches                         |
+------------------------------+---------+-------+---------------+
|          Endpoint            | Engine  | Port  |    Status     |
+------------------------------+---------+-------+---------------+
|  <VALKEY_ENDPOINT>           |  valkey |  6379 |  available    |
+------------------------------+---------+-------+---------------+
Valkey endpoint: <VALKEY_ENDPOINT>
```

ElastiCache **Serverless** matters here: KV cache traffic is bursty and its working-set size
is hard to predict. Serverless scales capacity automatically instead of forcing you to
right-size a node group for a cache whose demand you don't yet understand.

```bash
envsubst < 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml > /tmp/lmcache-valkey-pod.yaml
mv /tmp/lmcache-valkey-pod.yaml 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml

grep -E "model=s3://|rediss://" 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml

kubectl apply -f 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-configmap.yaml
kubectl apply -f 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml
```

```
        --model=s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/ \
      value: "rediss://<VALKEY_ENDPOINT>:6379"
configmap/valkey-cm created
pod/lmcache-valkey created
```

`rediss://` — double `s`, TLS. ElastiCache Serverless enforces encryption in transit. Using
`redis://` fails to connect, and the error is not obvious.

```bash
kubectl wait --for=condition=ready pod/lmcache-valkey --timeout=600s
kubectl logs lmcache-valkey | grep -E "ValkeyConnectorAdapter|Connected to remote storage"
```

```
pod/lmcache-valkey condition met
LMCache INFO: Discovered adapter: ValkeyConnectorAdapter
LMCache INFO: Connected to remote storage at rediss://<VALKEY_ENDPOINT>:6379,
              remote_mla_worker_id_as_0 mode: False
```

### Measure it

```bash
kubectl exec lmcache-valkey -- python /app/query-twice.py
```

```
============================================================
LMCache L1 (CPU RAM) + L2 (Valkey Remote) Cache Test
============================================================
Cold request writes KV tensors to BOTH L1 (local CPU RAM) and L2
(Valkey, shared across pods) in parallel. Warm request serves the
matching prefix from LMCache (L1 first, falling back to L2 on miss).

=== Cold request: LMCache empty on this pod, GPU computes all tokens ===
[OK] Cold TTFT: 2.907s (KV tensors written to L1 and L2 in parallel)

=== Warm request: same 90% content, LMCache should serve the matching prefix ===
[OK] Warm TTFT: 0.202s (GPU only computed the new 10% tail)

------------------------------------------------------------
TTFT Improvement: 2.704s (14.4x faster)
------------------------------------------------------------
```

Cold TTFT went 2.753 s → 2.907 s (+154 ms): the write-through cost of populating L2 as well as
L1. Warm TTFT went 0.132 s → 0.202 s. You pay a small, bounded latency tax on misses to buy
cross-pod, cross-restart durability.

### What the write path costs

```bash
kubectl logs lmcache-valkey | grep "Stored" | tail -3
```

```
LMCache INFO: [req_id=chatcmpl-aa41df…] Stored 4096 out of total 4096 tokens.
  size: 0.5312 GB, cost 48.1860 ms, throughput: 11.0250 GB/s;
  offload_time: 48.0227 ms, put_time: 0.1310 ms
LMCache INFO: [req_id=chatcmpl-b1b825…] Stored 512 out of total 512 tokens.
  size: 0.0664 GB, cost 6.1192 ms, throughput: 10.8521 GB/s;
  offload_time: 6.0377 ms, put_time: 0.0547 ms
```

**0.53 GB of KV tensors for a 4,096-token prompt** — ~133 KB per token. That is why the GPU's
28 GiB of KV cache only holds ~215,000 tokens, and why offloading to CPU RAM (hundreds of GB)
buys so much.

Also note `offload_time: 48.02 ms` vs `put_time: 0.13 ms`. The expensive part is moving
tensors GPU→CPU, not writing to Valkey. Optimize the transfer, not the network.

## Part 3 — prove the cache survives pod deletion

This is the actual test of L2. Count the keys, destroy the pod, count again.

```bash
kubectl run vcheck --rm -i --restart=Never --image=redis:7-alpine -- \
  redis-cli --tls -h $VALKEY_ENDPOINT -p 6379 dbsize
```

```bash
kubectl delete pod lmcache-valkey
```

```
pod "lmcache-valkey" deleted from default namespace
```

```bash
kubectl run vcheck --rm -i --restart=Never --image=redis:7-alpine -- \
  redis-cli --tls -h $VALKEY_ENDPOINT -p 6379 dbsize
```

```
# Output: 36 (unchanged, L2 persisted across pod deletion)
36
```

Now bring up a **completely fresh pod** — empty L1, no GPU cache, nothing local:

```bash
kubectl apply -f 400-lmcache/remote-cache-sharing-with-valkey/lmcache-valkey-pod.yaml
kubectl wait --for=condition=ready pod/lmcache-valkey --timeout=600s
kubectl logs lmcache-valkey | grep -E "ValkeyConnectorAdapter|Connected to remote storage"
kubectl exec lmcache-valkey -- python /app/query-twice.py
```

```
[OK] Cold TTFT: 4.687s (KV tensors written to L1 and L2 in parallel)
[OK] Warm TTFT: 0.400s (GPU only computed the new 10% tail)
------------------------------------------------------------
TTFT Improvement: 4.287s (11.7x faster)
------------------------------------------------------------
```

```bash
kubectl logs lmcache-valkey | grep -E "hit tokens|Retrieved" | head -4
```

```
LMCache INFO: [req_id=chatcmpl-a3d695…] Retrieved 4096 out of 4096 required tokens
  (from 4096 total tokens). size: 0.5312 gb, cost 2204.2588 ms, throughput: 0.2410 GB/s
LMCache INFO: [req_id=chatcmpl-88c231…] Retrieved 512 out of 512 required tokens
  (from 4096 total tokens). size: 0.0664 gb, cost 283.3166 ms, throughput: 0.2344 GB/s
```

**`Retrieved 4096 out of 4096 required tokens`** on a pod that had never seen the prompt. The
KV tensors were fetched from Valkey. This is the capability that makes L2 worth its complexity:

- A **new replica** scaled up by the HPA is warm immediately, not cold.
- A **rolling update** doesn't throw away accumulated cache.
- **Load balancing** is free — any pod can serve any user's follow-up turn.
- A **crashed pod** costs no cache.

Mind the throughput though: **0.24 GB/s from Valkey vs ~11 GB/s to CPU RAM — roughly 45×
slower.** Retrieving 0.53 GB took 2.2 seconds. L2 is still far cheaper than recomputing (which
would have cost 4.7 s here), but the hierarchy is real: keep hot prefixes in L1, use L2 for
breadth and durability.

```bash
kubectl delete pod lmcache-valkey
kubectl delete configmap valkey-cm
```

## Summary of trade-offs

| | L1 (CPU RAM) | L2 (Valkey) |
|---|---|---|
| Read throughput | ~11 GB/s | ~0.24 GB/s |
| Warm TTFT achieved | 0.132 s | 0.202 s (0.400 s on a cold pod) |
| Cold-path overhead | baseline | +154 ms write-through |
| Survives pod restart | ✗ | ✓ |
| Shared across replicas | ✗ | ✓ |
| Extra infrastructure | none | ElastiCache Serverless (hourly cost) |

Use L1 always — it's free and it's 20×. Add L2 when you run multiple replicas, autoscale, or
deploy often enough that discarding cache on every rollout hurts.

Next: [07 — LoRA fine-tuning](07-lora-fine-tuning.md)
