# Glossary

Every term and acronym used in this tutorial, in one place. For the *why* behind the important
ones, see [CONCEPTS.md](CONCEPTS.md) — this page is deliberately just definitions.

Jump: [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) ·
[K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) ·
[T](#t) · [U](#u) · [V](#v) · [W](#w)

---

## A

**Adapter** — The small file produced by LoRA fine-tuning, holding only the learned weight
deltas. 83.3 MB here, versus 10.4 GB for the base model. [Ch 07](07-lora-fine-tuning.md)

**ALB (Application Load Balancer)** — AWS layer-7 load balancer. Created automatically from a
Kubernetes `Ingress` by the AWS Load Balancer Controller. Three are used in this tutorial (Open
WebUI, Grafana, Gradio).

**AMI (Amazon Machine Image)** — The disk image a node boots from. EKS Auto Mode's GPU AMI
ships the NVIDIA driver and container runtime hooks pre-installed, so you never install a driver
by hand. [Ch 01](01-gpu-capacity.md)

**Attention** — The transformer operation where each token weighs every previous token. It's
what makes the [KV cache](#k) necessary.

**Auto Mode** — See [EKS Auto Mode](#e).

## B

**BF16 (bfloat16)** — 16-bit floating point, 2 bytes per number. 8.9 B parameters × 2 bytes ≈
17.8 GB of weights. The arithmetic that determines how much VRAM is left for the KV cache.

**Batching** — Processing multiple requests together so one read of the model weights serves all
of them. The source of nearly all inference throughput.
[Why](CONCEPTS.md#why-batching-is-the-whole-trick)

## C

**Chart (Helm)** — A packaged, templated set of Kubernetes manifests.
[Ch 05](05-benchmarking.md) ships the benchmark as a local chart.

**Chunking** — Splitting documents into passages small enough to embed and to fit in a prompt.
The quality ceiling of any RAG system. [Ch 08](08-rag-s3-vectors.md)

**ClusterIP** — The default `Service` type: reachable only from inside the cluster. `vllm-serve-svc`
is a ClusterIP; the outside world reaches it only via an Ingress.

**ConfigMap** — Non-secret configuration stored in the API and mounted into pods as files or
environment variables. Used for the LMCache config and the training script.

**Container runtime hook** — The shim that injects GPU devices and driver libraries into a
container when it requests `nvidia.com/gpu`. Pre-wired by the EKS Auto Mode GPU AMI.

**Context window** — The maximum tokens (prompt + output) one request may use. 8,192 here.
Directly trades off against concurrency: bigger window, fewer simultaneous users.

**Continuous batching** — vLLM adds and removes sequences from the running batch every
iteration, instead of waiting for a whole batch to finish. Explains the *Num Running* /
*Num Waiting* panels. [Ch 03](03-observability.md)

**Cosine distance / similarity** — How closely two embedding vectors point in the same
direction. Similarity 1.0 = identical, 0 = unrelated. **Distance = 1 − similarity, so lower is
closer.** [Why](CONCEPTS.md#part-9--embeddings-and-vector-search-intuitively)

**CRD (Custom Resource Definition)** — Adds a new object type to the Kubernetes API.
`NodePool`, `ServiceMonitor`, `GrafanaDashboard` and `RayService` are all CRDs.

**CSI (Container Storage Interface)** — The plugin standard for storage drivers. The Mountpoint
for Amazon S3 CSI driver makes an S3 bucket appear as a directory inside a pod.

## D

**DCGM (Data Center GPU Manager)** — NVIDIA's GPU management library. **DCGM Exporter**
republishes its readings as Prometheus metrics on port 9400: utilization, temperature, power,
framebuffer used. [Ch 03](03-observability.md)

**Decode** — The generation phase: one token per step, sequential, **memory-bandwidth-bound**.
Determines ITL. [Why](CONCEPTS.md#generation-happens-in-two-very-different-phases)

**Deployment** — Kubernetes object that keeps N identical pods running and replaces failures.
How vLLM runs in chapters 02–08.

**DLC (Deep Learning Container)** — An AWS-published image with a pre-solved
CUDA/framework/Ray version matrix. Used in [Ch 09](09-ray-serve.md) to avoid assembling
Ray + vLLM + CUDA yourself.

## E

**E2E latency** — Total wall-clock time for a complete response: TTFT + (ITL × output tokens).

**EKS (Elastic Kubernetes Service)** — AWS-managed Kubernetes.

**EKS Auto Mode** — EKS variant where AWS also manages the nodes, running Karpenter for you with
built-in `system` and `general-purpose` NodePools. This tutorial adds a `gpu` NodePool.
[Ch 00](00-environment-and-cluster.md)

**ElastiCache Serverless for Valkey** — Managed Redis-compatible cache with no instance sizing.
Serves as LMCache's L2 tier over `rediss://` (TLS). [Ch 06](06-kv-cache-offloading.md)

**Embedding** — A vector of numbers representing text's meaning, such that similar meanings land
nearby. 384 dimensions here, from `all-MiniLM-L6-v2`.

**ECR (Elastic Container Registry)** — AWS private container registry. Destination for the
[Ch 04](04-agentic-app-strands.md) agent image.

**Ephemeral storage IOPS** — Disk throughput on the node's local volume. Raised to 3,000 in the
GPU `NodeClass` so SOCI lazy image loading isn't disk-starved.
[Ch 01](01-gpu-capacity.md)

**`--enforce-eager`** — vLLM flag disabling CUDA graph capture. Saves startup time and memory;
costs some per-step latency. Used throughout, and a caveat worth remembering when reading the
benchmark numbers. [Ch 02](02-serve-with-vllm.md)

## F

**Framebuffer (FB)** — GPU memory. DCGM reports `FB_USED` / `FB_FREE` in MiB. 40,562 MiB used
at 0 % utilization is the tutorial's central cost lesson.

**FP8** — 8-bit floating point, 1 byte per number. This model's weights ship FP8 and are
upcast to BF16 for serving.

## G

**GCS (Global Control Store)** — Ray's cluster metadata service, on port 6379. One of the eight
ports a `RayService` exposes.

**Gradio** — Python library for quick ML web UIs. The RAG comparison app in
[Ch 08](08-rag-s3-vectors.md).

**grafana-operator** — Operator that reconciles `Grafana` and `GrafanaDashboard` custom
resources into a real Grafana instance. Makes dashboards version-controlled YAML.

## H

**HBM (High Bandwidth Memory)** — The GPU's on-package memory. Its **bandwidth** (~864 GB/s on
an L40S) sets the decode speed floor; its **capacity** (48 GB) sets your concurrency limit.

**Helm** — Kubernetes package manager. `helm install` is **not** idempotent — use
`helm upgrade --install`. [Why](CONCEPTS.md#part-7--helm-in-sixty-seconds)

## I

**Ingress** — Kubernetes object describing external HTTP routing. A controller turns it into a
real load balancer.

**IngressClassParams** — AWS LB Controller CRD holding cluster-wide ALB defaults, including
`inboundCIDRs`. **A per-Ingress `inbound-cidrs` annotation overrides it** — the reason two ALBs
in this tutorial end up open to `0.0.0.0/0`. [Ch 00](00-environment-and-cluster.md)

**inference-perf** — Open-source LLM load-testing harness, v0.6.1, packaged here as a local Helm
chart. [Ch 05](05-benchmarking.md)

**`ignoreEos`** — Benchmark setting that ignores the end-of-sequence token so every request
generates exactly the requested number of output tokens. Essential for comparable numbers.

**ITL (Inter-Token Latency)** — Time between consecutive streamed tokens. Perceived "typing
speed". Bounded by memory bandwidth and batch size. Compare [TPOT](#t).

## K

**Karpenter** — Kubernetes node autoscaler that provisions EC2 instances in response to
unschedulable pods, and removes them when unneeded. `NODES 0` on the GPU NodePool is the correct
idle state. [Why](CONCEPTS.md#part-5--karpenter-and-eks-auto-mode-nodes-that-appear-when-needed)

**KubeRay** — Operator that turns `RayCluster` / `RayService` CRs into running Ray clusters.
[Ch 09](09-ray-serve.md)

**kube-prometheus-stack** — Helm chart bundling Prometheus, the Prometheus Operator, Grafana and
Alertmanager. Its `release:` label is what `ServiceMonitor`s must carry to be discovered.

**KV cache** — Cached Key and Value attention tensors for tokens already processed, so they
aren't recomputed on every generation step. ~136 KB per token here; **it is the real capacity
limit of an inference server.**
[Why](CONCEPTS.md#the-kv-cache-why-memory-not-compute-is-your-capacity-limit)

## L

**L1 / L2 (LMCache tiers)** — L1 is CPU RAM on the same node (~11 GB/s); L2 is a shared remote
cache, here ElastiCache for Valkey (~0.24 GB/s). Both beat recomputing.
[Ch 06](06-kv-cache-offloading.md)

**L40S** — NVIDIA data-center GPU, 48 GB, ~864 GB/s memory bandwidth. The single card
everything in this tutorial runs on.

**Label / selector** — Arbitrary key/value tags, and the queries that match them. Kubernetes'
universal wiring mechanism — and, because **a selector matching nothing is not an error**, the
most common source of silent failure.
[Why](CONCEPTS.md#labels-and-selectors-the-one-idea-behind-most-silent-failures)

**LMCache** — Library that extends vLLM's KV cache into a memory hierarchy (GPU → CPU RAM →
remote store). Version 0.4.5 here. Requires `PYTHONHASHSEED=0` for stable cache keys.

**LoRA (Low-Rank Adaptation)** — Fine-tuning that freezes the base weights and learns two small
matrices per layer instead. 17.6 M trainable params (0.20 %), and roughly **90 MiB of extra
VRAM** to serve a variant. [Why](CONCEPTS.md#lora-without-the-math)

## M

**Ministral-3-8B-Instruct-2512** — The model served throughout: 8.9 B parameters, 10.4 GB on
disk as a single `consolidated.safetensors`.

**Mountpoint for Amazon S3** — Presents an S3 bucket as a filesystem. Its CSI driver
(`s3-csi-controller` / `s3-csi-node`) is how the LoRA adapter is read at serving time.

## N

**NodeClass / NodePool** — Karpenter CRDs. `NodePool` says *what kinds of nodes may be created*
(instance types, taints, limits); `NodeClass` says *how to build them* (AMI, disk, capacity
reservations). [Ch 01](01-gpu-capacity.md)

**`nodeSelector`** — A pod's requirement that it land on a node with certain labels. Note it is
**not** sufficient for GPU scheduling on its own — you also need a matching toleration.
[Why](CONCEPTS.md#getting-a-pod-onto-a-gpu-node-takes-two-separate-things)

**`NoSchedule` taint** — Marks a node as off-limits unless a pod explicitly tolerates it. How
GPU nodes avoid being occupied by CPU workloads.

**`nvidia.com/gpu`** — The resource name a pod requests to be allocated a physical GPU, and also
the taint key on GPU nodes.

**`nvidia-smi`** — NVIDIA's command-line GPU query tool. [Ch 01](01-gpu-capacity.md) runs it in
a throwaway pod to prove GPU scheduling works before deploying anything real.

**Num Running / Num Waiting** — vLLM metrics for sequences in the batch vs. queued. Running
plateauing while Waiting climbs **is** saturation. [Ch 03](03-observability.md)

## O

**ODCR (On-Demand Capacity Reservation)** — Reserved EC2 capacity guaranteeing a GPU is
available. **It bills even with zero instances running**, which is why
[Ch 10](10-cleanup.md) cancels it explicitly.

**OpenAI-compatible API** — vLLM's `/v1/completions`, `/v1/chat/completions`, `/v1/models`
endpoints, so any OpenAI client works unchanged. What lets Open WebUI, Strands and Gradio all
talk to a self-hosted model.

**Open WebUI** — Self-hosted chat UI, v0.11.0. Wired to vLLM through `OPENAI_API_BASE_URLS`.

**Operator** — A controller that watches custom resources and does the real work of realizing
them. Karpenter, Prometheus Operator, grafana-operator and KubeRay are all operators.

## P

**p50 / p95 / p99** — Percentiles. p99 means 1 request in 100 was worse than this value.
**Alert on tails, not means.** [Why](CONCEPTS.md#percentiles-briefly)

**PagedAttention** — vLLM's technique of storing the KV cache in fixed-size blocks, like OS
virtual-memory pages, to avoid fragmentation and enable prefix sharing.

**PEFT (Parameter-Efficient Fine-Tuning)** — Hugging Face library implementing LoRA and friends.
Version 0.19.1 here.

**Pod** — The smallest Kubernetes deployable unit: one or more containers sharing a network
address, scheduled together, and **disposable by design**.

**Pod Identity (EKS)** — Attaches an IAM role to a pod's service account, so no AWS access keys
appear in any manifest.

**PodMonitor** — Like a `ServiceMonitor` but scrapes pods directly. Needed in
[Ch 09](09-ray-serve.md) to keep Ray head and worker metrics separate.

**Prefill** — The phase that processes the whole prompt at once, in parallel.
**Compute-bound**, and what TTFT mostly measures.
[Why](CONCEPTS.md#generation-happens-in-two-very-different-phases)

**Prefix caching** — Reusing KV tensors when requests share a leading prompt. 0 % hit rate on
synthetic benchmarks ([Ch 05](05-benchmarking.md)); the entire basis of the 20× win in
[Ch 06](06-kv-cache-offloading.md).

**Prometheus** — Metrics system that **pulls** by scraping HTTP `/metrics` endpoints on a
schedule. [Why](CONCEPTS.md#part-6--how-the-metrics-actually-get-into-grafana)

**PromQL** — Prometheus' query language, what every Grafana panel here is written in.

**PV / PVC (PersistentVolume / PersistentVolumeClaim)** — Storage that outlives a pod, and a
pod's request for some. Used to mount S3 for the LoRA adapter.

**`PYTHONHASHSEED=0`** — Environment variable that makes Python's string hashing deterministic.
Without it, LMCache computes different cache keys in different processes and remote cache hits
silently never happen. [Ch 06](06-kv-cache-offloading.md)

## Q

**Queue time** — How long a request waits before the engine starts on it. Invisible in
throughput graphs, and the dominant term in TTFT once you pass the knee.

## R

**RAG (Retrieval-Augmented Generation)** — Embed the question, retrieve relevant passages, paste
them into the prompt. Changes *what the model knows*, not how it behaves.
[Ch 08](08-rag-s3-vectors.md)

**Ray / Ray Serve** — Distributed Python runtime, and its model-serving layer. Worth it for
multi-node models and multi-model pipelines; overkill for one model on one GPU.
[Ch 09](09-ray-serve.md)

**RayService** — KubeRay CRD describing a Ray cluster plus a Serve application. Exposes eight
Service ports, of which one (8000) is the API you actually call.

**Reconciliation** — The control loop at the heart of Kubernetes: continuously compare desired
state to actual state, and close the gap.

**`rediss://`** — Redis protocol over TLS. Required by ElastiCache Serverless — plain
`redis://` fails. [Ch 06](06-kv-cache-offloading.md)

**Run:ai Model Streamer (`runai_streamer`)** — vLLM load format that streams weights from object
storage into VRAM using many concurrent ranged GETs, instead of downloading to disk first.
Lets `--model=s3://…` work with no volume at all. [Ch 02](02-serve-with-vllm.md)

## S

**S3 Vectors** — Amazon S3 bucket type providing native vector storage and nearest-neighbour
queries, via `aws s3vectors`. Vector search with no database to run.
[Ch 08](08-rag-s3-vectors.md)

**safetensors** — Safe, memory-mappable tensor file format. This model is one 10.4 GB
`consolidated.safetensors`.

**Saturation knee** — The load level past which latency degrades sharply while throughput barely
improves. Measured at **8.17 req/s** here; **this, not peak throughput, is your capacity.**
[Why](CONCEPTS.md#why-latency-falls-off-a-cliff-instead-of-degrading-gently)

**Service** — Stable in-cluster DNS name and virtual IP, load-balancing to whichever pods match
its selector.

**ServiceMonitor** — Prometheus Operator CRD declaring a scrape target. **Must carry
`release: kube-prometheus-stack`** or it is created successfully and never read.
[Ch 03](03-observability.md)

**SOCI (Seekable OCI)** — Lazy container-image loading: start the container before the whole
image has been pulled. Why the GPU `NodeClass` raises ephemeral-storage IOPS.

**Strands Agents SDK** — AWS framework for building tool-using agents. [Ch 04](04-agentic-app-strands.md)
points one at the in-cluster vLLM to show an 8 B self-hosted model doing real tool calls.

**`success-codes: '200-302'`** — ALB annotation widening accepted health-check statuses. Needed
for apps whose root path redirects.

## T

**Taint / toleration** — A taint repels pods from a node; a toleration is a pod's opt-in. GPU
scheduling needs a toleration **and** a nodeSelector **and** a resource request.
[Why](CONCEPTS.md#getting-a-pod-onto-a-gpu-node-takes-two-separate-things)

**Tensor / pipeline parallelism** — Splitting one model across several GPUs or nodes. Not needed
here (the model fits on one card) — but it's the main legitimate reason to reach for Ray.

**Token** — The subword unit a model actually processes. ~0.75 words in English. Every limit,
price and metric in LLM serving is denominated in tokens.

**Tokenizer** — Maps text ↔ token IDs. Ships alongside a model but is a separate artifact, which
is why [Ch 05](05-benchmarking.md) has to point the benchmark at a compatible one.

**TPOT (Time Per Output Token)** — Average per-token generation time. Effectively the same
quantity as ITL, reported by a different tool.

**Transformer** — The architecture behind these models. For serving purposes: stacked layers of
attention plus feed-forward, whose weights get read from HBM on every decode step.

**TRL** — Hugging Face's training library for supervised fine-tuning and preference methods.
Version 1.5.1, used with PEFT for the LoRA run.

**TTFT (Time To First Token)** — Latency until the first output token appears. Prefill **plus
queue wait** — which makes it the first metric to collapse under overload, and the one users
feel.

## U

**Utilization (GPU)** — Percentage of time the GPU's compute units are busy. Reported by DCGM as
`DCGM_FI_DEV_GPU_UTIL`. **0 % while 40.6 GB is held and 115 W is drawn** is a fully-billed idle
GPU. [Ch 03](03-observability.md)

## V

**Valkey** — Open-source Redis fork. The engine behind ElastiCache Serverless here, used as
LMCache's L2 tier.

**Vector index** — The searchable structure over embeddings. `knowledge-base`, 384-dim, cosine,
25 vectors. [Ch 08](08-rag-s3-vectors.md)

**vLLM** — The inference engine used throughout, v0.21.0. Provides continuous batching,
PagedAttention, an OpenAI-compatible API, and a native Prometheus `/metrics` endpoint.

**VRAM** — GPU memory. A **hard** limit: exceed it and you get an OOM crash, not slow
degradation.

## W

**Warmup** — vLLM's startup profiling and cache allocation. Part of the 9.16 s
`init engine` measurement in [Ch 02](02-serve-with-vllm.md).

**Weights** — The model's learned parameters. 8.9 B of them, ~17.8 GB in BF16 — the number you
subtract from 48 GB to find your KV cache budget.

---

Back to: [README](README.md) · [CONCEPTS.md](CONCEPTS.md) · [GOTCHAS.md](GOTCHAS.md)
