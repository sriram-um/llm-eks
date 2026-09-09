# Concepts first: everything you need to follow this tutorial

You do not need to read this page in order, and you do not need to finish it before starting
Chapter 00. It exists so that when a chapter says *"KV cache utilization plateaued at 75 %,
which is why the knee is here"*, you already know what a KV cache is and why a plateau is the
interesting part.

Every chapter also opens with a short **"In plain terms"** box that links back here.

**If you know Kubernetes but not LLM serving,** read parts 1–3 and skip 4–7.
**If you know LLMs but not Kubernetes,** skip 1–3 and read 4–7.
**If you know neither,** read straight through — it's about 20 minutes.

---

## Part 1 — What a model actually does when you send it a prompt

### Text is not words, it's tokens

A language model never sees characters or words. Text is first cut into **tokens** — subword
chunks from a fixed vocabulary. Ministral-3-8B has a vocabulary of about 131,000 tokens.

```
"Serving LLMs on Kubernetes is fun"
   ↓ tokenizer
["Serv", "ing", " LL", "Ms", " on", " Kubernetes", " is", " fun"]
   ↓
[12457, 288, 27691, 4102, 402, 96331, 611, 4785]      ← what the GPU sees
```

Rules of thumb for English: **1 token ≈ 0.75 words ≈ 4 characters.** A 512-token prompt is
roughly 380 words, about one page. This matters because *everything* in inference is priced,
measured, and limited in tokens — never in words.

> The tokenizer is a separate artifact from the weights. Chapter 05 hits a real problem
> because the benchmark tool couldn't load *this model's* tokenizer and had to borrow a
> compatible one. Same vocabulary, different packaging.

### Generation happens in two very different phases

This is the single most useful thing to understand about LLM serving. A request is not one
uniform piece of work — it's two phases with opposite performance characteristics.

```
            ┌─────────────────── PREFILL ───────────────────┐  ┌──── DECODE ────┐
prompt →    │ process all 512 prompt tokens AT ONCE,        │  │ token 1        │
            │ in parallel, one big matrix multiply per layer│→ │ token 2        │
            │                                               │  │ token 3  …     │
            └───────────────────────────────────────────────┘  └────────────────┘
                              ▲                                        ▲
                    produces the FIRST token                  one token per step,
                    → this is your TTFT                       strictly sequential
                                                              → this is your ITL
```

**Prefill** reads the whole prompt at once. Every token can be processed in parallel, so the
GPU gets a big, dense matrix multiply — exactly what GPUs are built for. Prefill is
**compute-bound**: it's limited by raw FLOPs.

**Decode** generates one token, appends it to the sequence, then generates the next. It cannot
be parallelized across time, because token 5 depends on token 4. And here's the punchline: to
produce *one single token*, the GPU must read **every weight in the model** out of memory.

That's ~17.8 GB of weights for this model. An L40S has roughly 864 GB/s of memory bandwidth:

```
17.8 GB  ÷  864 GB/s  ≈  20 ms per decode step  →  ~48 tokens/sec for one request
```

Decode is **memory-bandwidth-bound**, not compute-bound. The GPU's arithmetic units sit mostly
idle while data moves. Look at Chapter 03's dashboard: at low load, *Time Per Output Token p99
was ~20 ms* — the back-of-envelope number above, landing almost exactly on the measurement.

### Why batching is the whole trick

If decode is limited by reading weights, and you have to read all the weights anyway... then
read them **once** and use them for 50 requests simultaneously. The weight-reading cost is
amortized across the batch.

That's why a server that does ~48 tokens/sec for one user can do ~2,400 tokens/sec across many
users (Chapter 05's measurement). **Throughput comes almost entirely from batching.**

vLLM uses **continuous batching**: instead of waiting for a batch of 32 requests to all finish
before starting the next batch, it adds and removes sequences from the running batch on every
single iteration. A request that finishes early frees its slot immediately.

This is the direct explanation for the most important panel in Chapter 03:

- **Num Running** — sequences currently in the batch. This has a hard ceiling.
- **Num Waiting** — sequences that arrived but couldn't fit. This is pure queueing.

When Num Running plateaus and Num Waiting climbs, you are saturated. No amount of extra
traffic will help.

### The KV cache: why memory, not compute, is your capacity limit

Inside the attention mechanism, every token looks at every previous token. To do that, each
token contributes a **Key** and a **Value** vector at every layer.

Naively, generating token 500 would recompute the K and V for tokens 1–499 all over again — and
token 501 would do it again. That's quadratic waste. So you cache them. That cache is the
**KV cache**, and it is the reason LLM serving is a memory-capacity problem.

The cache grows **linearly with every token, for every concurrent request**:

```
KV cache bytes  ≈  2 (K and V) × layers × kv_heads × head_dim × bytes_per_number × tokens
```

You don't need to compute that by hand, because vLLM prints the answer at startup, and this
tutorial measures it twice independently:

| Source | Measurement | Per token |
|---|---|---|
| Chapter 02 — vLLM startup log | 27.91 GiB holds 215,152 tokens | **~136 KB** |
| Chapter 06 — LMCache offload log | 0.53 GB for a 4,096-token prompt | **~130 KB** |

Two completely different subsystems agreeing on ~130 KB per token. Now the capacity arithmetic
falls out, and it's the most important calculation in this tutorial:

```
48 GB card
 − 17.8 GB  model weights (8.9 B parameters × 2 bytes for BF16)
 ─────────
 ≈ 28 GiB   left for KV cache
 ÷ 136 KB   per token
 ─────────
 = 215,152  tokens of total capacity, shared by ALL in-flight requests
 ÷ 8,192    tokens if each request uses its full context
 ─────────
 = 26.26    concurrent requests. ← vLLM printed exactly this: "Maximum concurrency: 26.26x"
```

**Your concurrency limit is a division problem, and the engine tells you the answer for free at
boot.** Chapter 05 then measures what happens when you exceed it. The numbers agree.

Two consequences that surprise people:

- **A longer context window costs concurrency.** Doubling max context halves how many users fit.
- **Memory is claimed up front, not on demand.** Chapter 03's dashboard shows 40.6 GB held at
  **0 % GPU utilization**. The card is reserved from the moment the model loads, whether or not
  anyone is using it. That's the economics of GPU inference in one screenshot.

> **Prefix caching** is the natural follow-on. If two requests start with the same text — a
> shared system prompt, the history of a chat conversation, a fixed RAG instruction block —
> their KV tensors for that shared part are *identical*, so they can be reused instead of
> recomputed. Chapter 05 shows a 0 % hit rate (synthetic prompts share nothing). Chapter 06
> engineers a 20× speedup out of exploiting it.

---

## Part 2 — The four numbers, and how to read a percentile

Almost every measurement in this tutorial is one of these four.

| Metric | Plain meaning | What it's bound by | Who cares |
|---|---|---|---|
| **TTFT** (Time To First Token) | How long until text starts appearing | prefill + **queue wait** | Anyone watching a screen |
| **ITL** / **TPOT** (Inter-Token Latency / Time Per Output Token) | Gap between successive tokens — the "typing speed" | memory bandwidth, batch size | Streaming UX |
| **Throughput** (tokens/sec, req/sec) | Total work the server completes | batching efficiency | Whoever pays the GPU bill |
| **E2E latency** | Total wall-clock for the whole response | TTFT + ITL × output length | Batch/offline jobs |

Intuition for what's "good": TTFT under ~300 ms feels instant. ITL of 20–50 ms reads as fast
typing. TTFT of 27 seconds — Chapter 05's saturated stage — means the user has closed the tab.

**TTFT is the metric that betrays you**, because it includes queue time. A saturated server
still has healthy-looking throughput and ITL while TTFT collapses. That's exactly the trap
Chapter 05 documents: pushing from 8 to 9 req/s bought **+7 % throughput** and **114× worse
TTFT**.

### Percentiles, briefly

You'll see `p50`, `p95`, `p99`.

- **p50** (median) — half of requests were faster than this.
- **p95** — 95 % were faster; 1 in 20 was worse.
- **p99** — 1 in 100 was worse.

Averages hide tails, and tails are what users complain about. If p50 is 200 ms and p99 is
30 s, then one request in a hundred is a disaster — and a user who makes 100 requests in a
session hits it roughly once. **Always alert on p95/p99, never on the mean.**

### Why latency falls off a cliff instead of degrading gently

This is queueing theory, and one sentence of it explains the whole shape of Chapter 05.

If requests arrive faster than the server can finish them, the queue does not settle at some
larger-but-stable length. It **grows without bound for as long as the overload lasts**. Waiting
time grows with it.

```
offered rate < capacity   →  queue stays short, latency ≈ service time      (stable)
offered rate ≈ capacity   →  queue becomes sensitive, latency climbs fast   (the knee)
offered rate > capacity   →  queue grows for as long as you keep pushing    (unstable)
```

So **throughput degrades gracefully and latency degrades catastrophically.** Two rules follow,
and they're the practical payoff of this whole section:

1. Find the knee and treat it as your capacity — not the peak-throughput point.
2. Autoscale on queue depth or TTFT, not on tokens/sec.

---

## Part 3 — Why a GPU, and why utilization is the entire cost story

A CPU has a few dozen fast, general-purpose cores. A GPU has thousands of simple cores plus
very wide memory. Matrix multiplication — which is essentially all a transformer does — is the
ideal workload for that shape.

But GPUs are **rented whole**. You cannot buy 30 % of an L40S. So the cost model is brutally
simple:

```
your cost  =  (hours the GPU exists)  ×  (price per hour)
```

Note what's missing: how much work you did. Chapter 03's screenshot — **0 % utilization,
40.6 GB held, 115 W drawn** — is a GPU costing full price to do nothing.

That single fact is why the middle chapters of this tutorial exist:

| Chapter | Lever | Effect |
|---|---|---|
| 05 Benchmarking | Know your real capacity | Don't over- or under-provision |
| 06 KV caching | Don't recompute what you've already computed | 20× faster TTFT on warm prefixes |
| 07 LoRA | Serve N variants on one card | ~90 MiB per variant instead of another GPU |

A second thing to internalize: **VRAM is a hard wall, not a soft one.** Exceed CPU RAM and
Linux swaps and gets slow. Exceed VRAM and you get an out-of-memory crash. There's no graceful
degradation, which is why the arithmetic in Part 1 matters before you deploy, not after.

---

## Part 4 — Kubernetes, only the parts this tutorial uses

Skip this if you already run workloads on Kubernetes.

The core idea is **declarative reconciliation**. You don't issue commands like "start this
container." You submit a document describing the desired state, and controllers continuously
work to make reality match it. If a pod dies, nobody has to notice — the controller sees a gap
between desired and actual, and closes it.

```bash
kubectl apply -f thing.yaml    # "here is what I want to be true"
```

### The objects you'll meet

| Object | What it is | Where you'll see it |
|---|---|---|
| **Pod** | One or more containers scheduled together on one node, sharing a network address. The smallest deployable unit — and **disposable by design**. | The vLLM server, the LMCache experiments |
| **Deployment** | "Keep N identical pods running." Replaces failures, handles rolling updates. | `mistral`, `open-webui`, `rag-service` |
| **Job** | "Run this to completion." On failure it **creates a new pod** and leaves the failed one behind. | model download, LoRA training, benchmarks |
| **Service** | A stable in-cluster DNS name and virtual IP that load-balances to whichever pods currently match its label selector. | `vllm-serve-svc:8000` |
| **Ingress** | HTTP routing from the outside world. A controller turns it into a real cloud load balancer. | The three ALBs |
| **ConfigMap** / **Secret** | Configuration and credentials injected as files or environment variables. | LMCache config, training scripts, Grafana password |
| **PersistentVolume / Claim** | Storage that outlives a pod. | Mounting the S3 bucket to read the LoRA adapter |
| **Namespace** | A folder for scoping names. | `default` for apps, `monitoring` for observability |

> That Job behaviour explains the tutorial's very first surprise. Chapter 00 shows a
> `model-download` pod in `Init:Error` *next to* one that `Completed`, while the Job itself says
> `Complete 1/1`. Nothing is broken: attempt one failed, the retry succeeded, and Kubernetes
> deliberately kept the corpse for you to inspect. **Judge a Job by the Job, not by its pods.**

### Labels and selectors: the one idea behind most silent failures

Kubernetes objects almost never reference each other by name. They match on **labels** —
arbitrary key/value tags — using **selectors**.

```yaml
# A Service doesn't say "send traffic to pod mistral-abc123".
# It says "send traffic to whatever currently carries this label":
selector:
  model: mistral
```

This is what makes the system resilient: pods come and go, and the Service just keeps matching
whatever exists. But it has a sharp edge. **A selector that matches nothing is not an error.**
It's a valid configuration that quietly does nothing.

Read the [GOTCHAS](GOTCHAS.md) file with that lens and a pattern jumps out: a `ServiceMonitor`
missing one label is never scraped; Open WebUI pointing at a Service name that doesn't exist
comes up perfectly healthy with zero models. **6 of the 14 documented failures produced no
error message at all**, and label/name mismatches are the largest family among them.

### Getting a pod onto a GPU node takes two separate things

This trips up nearly everyone, because it needs two mechanisms that sound redundant but aren't.
They point in opposite directions:

```yaml
nodeSelector:                       # ← YOU choose the node ("put me on a GPU node")
  karpenter.sh/nodepool: gpu

tolerations:                        # ← the NODE admits you ("I'll accept GPU workloads")
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"

resources:
  limits:
    nvidia.com/gpu: 1               # ← the DEVICE PLUGIN hands you a physical GPU
```

A **taint** on a node repels pods — it's how GPU nodes stop a random web frontend from squatting
on $2/hour hardware. A **toleration** is a pod's opt-in to that taint. A **nodeSelector** is the
pod's own preference.

Miss the nodeSelector and you land on a CPU node. Miss the toleration and you stay `Pending`
forever. That's why [Chapter 01](01-gpu-capacity.md) proves all three work using a throwaway
pod that just runs `nvidia-smi`, *before* asking a 10 GB model server to schedule.

### Operators and CRDs: why dashboards are YAML here

A **Custom Resource Definition** adds a new object type to the Kubernetes API. An **operator**
is a controller that watches those objects and does the real work.

Once you see it, half this tutorial's stack is the same idea repeated:

| Custom resource | Operator | What it reconciles |
|---|---|---|
| `NodePool` / `NodeClass` | Karpenter | EC2 instances |
| `ServiceMonitor` / `PodMonitor` | Prometheus Operator | Prometheus scrape config |
| `GrafanaDashboard` | grafana-operator | Dashboards inside Grafana |
| `RayService` | KubeRay | A whole Ray cluster |
| `Ingress` | AWS LB Controller | An Application Load Balancer |

The payoff is in [Chapter 03](03-observability.md): dashboards are version-controlled YAML that
gets recreated automatically if Grafana is replaced. Nobody clicks "export JSON" from a browser.
And [Chapter 09](09-ray-serve.md) swaps the entire serving layer for Ray while the monitoring
stack keeps working — total cost, two `PodMonitor` files.

---

## Part 5 — Karpenter and EKS Auto Mode: nodes that appear when needed

Classic Kubernetes assumes a pool of machines already exists. **Karpenter** inverts that: it
watches for pods that *cannot be scheduled* and buys hardware to fit them.

```
1. You create a pod needing nvidia.com/gpu: 1
2. No node can satisfy it → pod is Pending
3. Karpenter reads the NodePool, picks an instance type, launches an EC2 instance
4. Node joins the cluster (~1–2 min), plus image pull time
5. Pod schedules
       … and when the pod goes away, Karpenter removes the node again
```

**EKS Auto Mode** is AWS running that machinery for you, with `system` and `general-purpose`
NodePools built in. This tutorial adds a third, `gpu`.

Three consequences that show up in the chapters:

- **`NODES 0` is the correct steady state.** Chapter 01 opens with a GPU NodePool that has no
  nodes. You aren't billed for a GPU that doesn't exist yet.
- **Cold starts are real.** Node provisioning plus a multi-GB image pull plus loading 10.4 GB of
  weights is minutes, not seconds. This is the entire explanation for Chapter 09's
  `kubectl wait` timing out at 10 minutes.
- **GPU capacity can simply be unavailable.** L40S/H100/A100 supply is genuinely scarce.
  Chapter 01 uses an **On-Demand Capacity Reservation** so Karpenter is guaranteed to get a
  card — which also means you pay for the reservation whether you use it or not, which is why
  Chapter 10 is careful to cancel it.

---

## Part 6 — How the metrics actually get into Grafana

Prometheus **pulls**. Your application does not push metrics anywhere; it exposes a plain-text
HTTP endpoint, and Prometheus fetches it on a schedule.

```
vLLM pod  ──/metrics──►  Prometheus (scrapes every 30s)  ──►  Grafana (queries with PromQL)
DCGM pod  ──/metrics──►
```

vLLM ships `/metrics` natively — no sidecar, no exporter. For the GPU hardware itself you need
**DCGM Exporter**, which reads NVIDIA's management library and republishes it in the same format.

You need both, because they answer different questions, and neither answers the other's:

- **DCGM** = hardware truth: temperature, power, memory held, utilization.
- **vLLM `/metrics`** = software truth: queue depth, TTFT, cache hit rate, batch size.

A GPU at 95 % utilization tells you nothing about whether requests are queueing for 30 seconds.

A **ServiceMonitor** is the CRD that tells Prometheus what to scrape. And it contains this
tutorial's most instructive silent failure:

```yaml
metadata:
  labels:
    release: kube-prometheus-stack   # ← the Prometheus Operator's OWN selector
```

Prometheus Operator only picks up `ServiceMonitor`s carrying that label. Omit it and your
`ServiceMonitor` is created successfully, looks completely correct in `kubectl get`, and is
never read by anything. Selectors again — see Part 4.

---

## Part 7 — Helm in sixty seconds

**Helm** is a package manager for Kubernetes. A **chart** is a bundle of templated manifests; a
**release** is one installed instance of a chart; `values.yaml` holds the knobs.

```bash
helm install my-thing some-repo/some-chart -f my-values.yaml
```

Two things worth knowing before you hit them:

- **`helm install` is not idempotent.** Run it twice and you get
  `cannot re-use a name that is still in use`. Use `helm upgrade --install` in anything you
  might re-run. Chapter 09 hits this.
- **Charts are a fine way to ship a load test.** Chapter 05 packages `inference-perf` as a local
  chart specifically so a benchmark is a reviewable, versioned artifact instead of a script on
  somebody's laptop.

---

## Part 8 — Three ways to change what a model does

Chapters 06, 07 and 08 are each a different answer to "the model isn't doing what I need."
Choosing wrong wastes a lot of time, and the distinction is genuinely simple:

| Approach | Changes | Cost | Use when |
|---|---|---|---|
| **Prompting** | Nothing — you just ask better | free, instant | Always try first |
| **RAG** (Ch 08) | *What the model knows* | one retrieval hop, ~220 ms | Facts change, are private, or must be cited |
| **Fine-tuning / LoRA** (Ch 07) | *How the model responds* | minutes of GPU + a small artifact | You need a consistent format, tone, or structure |

The mnemonic: **RAG for knowledge, fine-tuning for behaviour.**

If your model invents product specs, that's a knowledge problem — RAG. If it rambles when you
need six fixed headings, that's a behaviour problem — fine-tune. Chapter 07 makes this explicit:
200 examples cannot teach the model new facts about venture capital, but they teach a rigid
output skeleton extremely cheaply.

### LoRA, without the math

Fine-tuning normally means updating all 8.9 billion weights: you need optimizer state for every
one, which is far more memory than the model itself, and you end up with a second 10.4 GB model
to store and serve.

**LoRA** (Low-Rank Adaptation) freezes the original weights and learns a small "diff" instead.
For each big weight matrix it trains two skinny matrices whose product has the same shape:

```
frozen:  W  (e.g. 4096 × 4096  = 16.8 M numbers)
learned: A  (16 × 4096)  and  B  (4096 × 16)   = 131 K numbers   ← ~1.3 % of the size

at inference:  output = W·x  +  B·(A·x)
                        ▲            ▲
                    unchanged    tiny correction
```

The measured result in Chapter 07: **17.6 M trainable parameters, 0.20 % of the model, an 83 MB
adapter, 169 seconds of training.**

The serving consequence is the good part. Because the base weights are untouched, vLLM keeps
**one** copy of the model in VRAM and layers adapters on top — so a second "model" in the API
costs about **90 MiB**, not another 48 GB card. Ten specialized variants on one GPU is a normal
thing to do.

---

## Part 9 — Embeddings and vector search, intuitively

RAG needs a way to find text that is *about* the same thing as your question, even with no words
in common. That's what an **embedding model** provides: it maps text to a list of numbers — a
**vector** — such that similar meaning lands nearby.

This tutorial uses `all-MiniLM-L6-v2`, which produces **384 numbers** per chunk of text. Think of
each as a coordinate in 384-dimensional space, where direction encodes meaning.

```
"laptop battery life"    → [0.02, -0.41, 0.88, …]  ┐ close together
"how long does it charge"→ [0.04, -0.38, 0.91, …]  ┘
"speaker sound quality"  → [0.77,  0.12, -0.03, …] ← far away
```

Closeness is measured by the **angle** between vectors, not the distance between their tips —
because what matters is direction, not magnitude.

- **Cosine similarity**: 1.0 = identical direction, 0 = unrelated, −1 = opposite.
- **Cosine distance** = 1 − similarity. **Lower is closer.** This flip catches everyone once.

Chapter 08 shows both ends of the scale, and the contrast is the lesson:

| Query | Result | Meaning |
|---|---|---|
| A hand-made vector of all `0.1`s | distances ~0.978–0.981 | ~Orthogonal to everything. The index works; the *ranking is meaningless* |
| A real embedded question | similarities 0.567 / 0.546 / 0.526 | Genuine semantic matches |

Why the fake vector scores ~0.98 against every document: real embeddings spread positive and
negative values across all 384 dimensions, so a uniformly-positive vector correlates with none
of them. Similarity ≈ 0, therefore distance ≈ 1. **A query that returns results is not evidence
that retrieval works.** You have to look at the scores.

The full RAG loop:

```
1. INGEST  (once, offline)
   documents → chunk → embed → store vectors + metadata

2. QUERY   (per request)
   question → embed → find nearest vectors → paste their text into the prompt → ask the LLM
```

Note what a "vector database" actually has to do: store vectors and find nearest neighbours.
Chapter 08 uses **Amazon S3 Vectors**, where that's a bucket type — no cluster to size or patch.
The measured retrieval overhead was **220 ms on a 15-second request: 1.4 %.**

**Retrieval is essentially free. Generation is the cost.** So the interesting question is never
"is RAG fast enough" — it's whether your chunks are any good.

---

## Part 10 — The AWS pieces, one line each

| Service | Role here | Why it's used |
|---|---|---|
| **EKS** | Managed Kubernetes control plane | Auto Mode also manages the nodes |
| **S3** | Model weights, datasets, benchmark results, LoRA adapters | vLLM reads weights *directly* from S3 — no volume to provision |
| **ECR** | Private container registry | Where the Chapter 04 agent image lands |
| **ALB** | Internet-facing HTTP load balancer | Created automatically from an `Ingress` |
| **ElastiCache Serverless (Valkey)** | Shared KV-cache tier (L2) | Cache traffic is bursty; serverless avoids guessing a size |
| **S3 Vectors** | Vector store for RAG | Vector search without operating a database |
| **ODCR** | Guaranteed GPU capacity | L40S is scarce; **bills even at zero instances** |
| **EKS Pod Identity** | Gives a pod an IAM role | No access keys in any manifest — check the YAML, there are none |

Two AWS-specific traps this tutorial hits, both worth knowing in advance:

- **An ALB's DNS name exists before the ALB works.** `kubectl get ingress` returns a hostname
  the moment the load balancer is *created*, but provisioning and target registration take
  longer. Hence `aws elbv2 wait load-balancer-available` — the difference between a demo and
  60 seconds of `503` while you refresh and doubt yourself.
- **Health checks must point at something that returns 200.** Grafana's `/` redirects, so the
  correct target is `/api/health`. Get this wrong and the target group flaps unhealthy forever
  while the app is perfectly fine.

---

## Part 11 — The whole system on one page

```
                        ┌──────────── YOUR LAPTOP ────────────┐
                        │  kubectl / helm / aws  +  a browser │
                        └───────────────┬─────────────────────┘
                                        │  (ALBs locked to your IP — Chapter 00)
   ┌────────────────────────────────────▼──────────────────────────────────────┐
   │  EKS Auto Mode cluster                                                    │
   │                                                                           │
   │  system pool          general-purpose pool        gpu pool (L40S)         │
   │  ┌──────────────┐     ┌────────────────────┐     ┌──────────────────────┐ │
   │  │ Prometheus   │◄────┼─ Open WebUI        │     │ vLLM  ◄─ weights ────┼─┼─► S3
   │  │ Grafana      │     │  Gradio RAG UI     │────►│  ├ KV cache (28 GiB) │ │
   │  │ operators    │     │  Strands agent     │     │  ├ LoRA adapter      │ │
   │  └──────▲───────┘     │  benchmark Job     │     │  └ /metrics ─────────┼─┘
   │         │             └────────────────────┘     │ DCGM Exporter ───────┼──┐
   │         └───────────────── scrape /metrics ──────┴──────────────────────┘  │
   │                                                                            │
   └────────┬──────────────────┬───────────────────────┬────────────────────────┘
            │                  │                       │
      ElastiCache        S3 Vectors              S3 (results,
      (L2 KV cache)      (RAG index)              datasets, adapters)
```

Read the chapters as a progression through that diagram: get a GPU (01), put a model on it
(02), see what it's doing (03), let an app use it (04), find its limits (05), then three
different ways to do better — cache (06), adapt (07), retrieve (08) — then an alternative
serving layer (09), and finally turn it all off (10).

---

## If you remember only five things

1. **Prefill is compute-bound, decode is memory-bound.** That one split explains TTFT vs ITL,
   why batching produces almost all your throughput, and why caching prefixes wins so big.
2. **KV cache capacity is your concurrency limit**, and it's a division problem the engine
   solves for you at startup: 215,152 tokens ÷ 8,192 per request = 26.26 concurrent requests.
3. **Peak throughput and acceptable latency are different operating points.** Find the knee.
   Past it, latency degrades catastrophically while throughput barely improves.
4. **You pay for the whole GPU from the moment weights load** — 0 % utilization, 40.6 GB held,
   115 W. Utilization is a business metric.
5. **In Kubernetes, a selector that matches nothing is not an error.** It's the most common way
   for something to be completely broken and look completely fine.

---

Ready: [00 — Environment and cluster](00-environment-and-cluster.md) ·
Terms only: [GLOSSARY.md](GLOSSARY.md) ·
Things that went wrong: [GOTCHAS.md](GOTCHAS.md)
