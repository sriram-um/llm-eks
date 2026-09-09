# Running LLMs on Amazon EKS: An End-to-End Tutorial

A complete, field-tested walkthrough of serving, monitoring, benchmarking, optimizing,
fine-tuning and extending an open-weights LLM on **Amazon EKS Auto Mode** with **NVIDIA
L40S GPUs**.

Every command, manifest, and number in this tutorial comes from one real ~5-hour run on a
live cluster. The dashboard captures in [`screenshots/`](screenshots/) are from that same run.

Code: **https://github.com/sriram-um/llm-eks**

---

## Start here

You do **not** need prior LLM-serving experience. Two supporting files exist so that no chapter
depends on jargon you haven't met:

- **[CONCEPTS.md](CONCEPTS.md)** — a plain-language primer with diagrams: what a model actually
  does with your prompt, why the KV cache decides how many users fit on a GPU, what TTFT means
  and why it collapses first, and the ten Kubernetes objects this tutorial uses. ~20 minutes.
- **[GLOSSARY.md](GLOSSARY.md)** — every term and acronym, alphabetically, one line each.

Every chapter also opens with an **"In plain terms"** box giving the intuition before the
commands start.

Pick a path:

| If you are… | Read this |
|---|---|
| **New to both LLM serving and Kubernetes** | [CONCEPTS.md](CONCEPTS.md) end to end, then chapters 00 → 10 in order |
| **A Kubernetes user, new to LLMs** | [CONCEPTS Parts 1–3](CONCEPTS.md#part-1--what-a-model-actually-does-when-you-send-it-a-prompt), then 02 → 03 → 05 → 06 |
| **An ML engineer, new to Kubernetes** | [CONCEPTS Parts 4–7](CONCEPTS.md#part-4--kubernetes-only-the-parts-this-tutorial-uses), then 00 → 01 → 02 |
| **Here for one specific thing** | Jump straight to the chapter; follow the *Background* links in its intro box if a term is unfamiliar |
| **Debugging something that's broken** | [GOTCHAS.md](GOTCHAS.md) — 14 real failures, 6 of which produced no error message at all |

> **Read this before you deploy anything:** [Chapter 00](00-environment-and-cluster.md) locks the
> load balancer to your own IP. Two later chapters deliberately override that lock, and both say
> so. An unauthenticated chat UI on a public ALB is somebody else's free GPU.

## What you build

```
                          ┌──────────────────── Amazon EKS Auto Mode (v1.36) ────────────────────┐
                          │                                                                      │
  ALB (IP-locked)  ──────►│  Open WebUI ──┐                                                      │
  ALB (IP-locked)  ──────►│  Gradio RAG UI├──► vllm-serve-svc :8000 ──► vLLM 0.21.0  ┐           │
  ALB (IP-locked)  ──────►│  Grafana      │    (OpenAI-compatible)      Ministral-3-8B│  GPU     │
                          │               │                            + LoRA adapter │ NodePool │
                          │  Strands agent┘                            + LMCache      │ (L40S)   │
                          │                                                            ┘          │
                          │  Prometheus ◄── DCGM Exporter :9400  ◄── GPU node                    │
                          │       ▲                                                              │
                          │       └── ServiceMonitor (vLLM /metrics)                             │
                          │                                                                      │
                          │  inference-perf Job (Helm)   KubeRay RayService   LoRA training Job  │
                          └───────┬──────────────────┬───────────────────┬───────────────────────┘
                                  │                  │                   │
                       s3://…/benchmarks     ElastiCache Valkey    s3://…/anyvc-startup-lora
                                                  (L2 KV cache)
                          S3 Vectors bucket ──► index: knowledge-base (384-dim, cosine)
                          s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/ (10.4 GB)
```

## Chapters

| # | Chapter | What it covers |
|---|---------|----------------|
| 📖 | [Concepts first](CONCEPTS.md) | **Read if any term below is new.** Tokens, prefill vs decode, KV cache, TTFT/ITL, GPU economics, Kubernetes objects, Karpenter, Prometheus, LoRA, embeddings |
| 🔤 | [Glossary](GLOSSARY.md) | Every term and acronym, alphabetically |
| 00 | [Environment and cluster](00-environment-and-cluster.md) | EKS Auto Mode, Karpenter NodePools, Mountpoint-S3 CSI, pre-staged model download Job, IP-locking the ALB |
| 01 | [GPU capacity](01-gpu-capacity.md) | Custom GPU `NodeClass` with an EC2 Capacity Reservation, `nvidia-smi` smoke test, taints/tolerations |
| 02 | [Serve the model with vLLM](02-serve-with-vllm.md) | Loading a 10.4 GB model straight from S3 with `runai_streamer`, the OpenAI-compatible API, Open WebUI behind an ALB |
| 03 | [Observability](03-observability.md) | DCGM Exporter, `ServiceMonitor` for vLLM, Grafana via the grafana-operator — with real dashboard screenshots |
| 04 | [Agentic app with Strands](04-agentic-app-strands.md) | Build → ECR → deploy a tool-using agent whose model endpoint is the in-cluster vLLM |
| 05 | [Benchmarking](05-benchmarking.md) | `inference-perf` as a Helm chart, four load scenarios, finding the saturation knee (real numbers) |
| 06 | [KV cache offloading](06-kv-cache-offloading.md) | LMCache L1 (CPU RAM) and L2 (ElastiCache Serverless for Valkey) — **up to 20.9× faster TTFT** |
| 07 | [LoRA fine-tuning](07-lora-fine-tuning.md) | Train a 17.6 M-parameter adapter on-cluster in 169 s, serve it alongside the base model |
| 08 | [RAG with S3 Vectors](08-rag-s3-vectors.md) | Amazon S3 Vectors as the vector store, document-processor Job, side-by-side RAG comparison UI |
| 09 | [Ray Serve on EKS](09-ray-serve.md) | KubeRay `RayService`, the Ray dashboard, Ray + vLLM metrics into Prometheus |
| 10 | [Cleanup](10-cleanup.md) | Tearing everything down in the right order, including the capacity reservation |
| ⚠️ | [Gotchas](GOTCHAS.md) | **Every error hit during the real run and how it was resolved** |

## The stack, exactly as run

| Component | Version / value |
|---|---|
| EKS | Auto Mode, `v1.36.2-eks-0690643` |
| Region | `us-east-2` |
| GPU | 1 × NVIDIA **L40S** (48 GB), `karpenter.sh/nodepool=gpu` |
| Inference engine | vLLM **0.21.0** |
| Model | `Ministral-3-8B-Instruct-2512` (8.9 B params, FP8 weights → BF16) |
| Model storage | S3, loaded with `--load-format=runai_streamer` |
| Chat UI | Open WebUI `v0.11.0` |
| Metrics | kube-prometheus-stack + grafana-operator + NVIDIA DCGM Exporter |
| Benchmark harness | `inference-perf` `v0.6.1` |
| KV cache | LMCache `0.4.5` → CPU RAM (L1) + ElastiCache Serverless for Valkey (L2) |
| Fine-tuning | transformers `5.11.0`, TRL `1.5.1`, PEFT `0.19.1`, accelerate `1.13.0` |
| Vector store | Amazon S3 Vectors (`aws s3vectors`), 384-dim, cosine |
| Alternative serving | KubeRay `1.1.0` + Ray Serve |

## Headline results from the run

| Measurement | Value |
|---|---|
| GPU KV cache available after loading the 8.9 B model | **27.91 GiB** → 215,152 tokens |
| Engine init (profile + KV cache + warmup) | **9.16 s** |
| Peak sustainable throughput before the knee | **~2,100 output tok/s @ 8.17 req/s, TTFT p50 242 ms** |
| Past the knee (rate 20) | 2,253 output tok/s but **TTFT p50 27.5 s** — throughput +7 %, latency ×114 |
| LMCache L1 (CPU RAM) warm TTFT | 2.753 s → **0.132 s (20.9× faster)** |
| LMCache L2 (Valkey) warm TTFT, fresh pod | 4.687 s → **0.400 s (11.7× faster)** |
| LoRA adapter | 17.6 M trainable params (0.20 %), loss 0.1811, **169 s**, 83.3 MB |
| RAG retrieval | 25 vectors indexed, top-3 cosine similarity 0.567 / 0.546 / 0.526 |
| RAG latency overhead | 15.30 s → 15.52 s — **retrieval costs ~220 ms (1.4 %)** |
| Adapter VRAM cost when served with the base model | 27.91 → 27.82 GiB KV cache, i.e. **~90 MiB** |

## Prerequisites

- An EKS **Auto Mode** cluster (v1.31+) with the AWS Load Balancer Controller and the
  Mountpoint for Amazon S3 CSI driver installed.
- An **EC2 On-Demand Capacity Reservation** for a GPU instance type (this run used an ODCR
  referenced by ID from the GPU `NodeClass`). L40S capacity is scarce; without a
  reservation Karpenter may never get a node.
- `kubectl`, `helm`, `aws` CLI (with `s3vectors`), `envsubst`, `jq`, `docker`.
- An S3 bucket holding the model weights, and an **EKS Pod Identity** service account
  (`model-storage-sa`) with read/write on it. Chapter 08 additionally needs an `s3-access-sa`
  service account with `s3vectors:PutVectors` / `QueryVectors`, plus a vector bucket and a
  `knowledge-base` index.

> **Cost warning.** A GPU node, three ALBs, an ElastiCache Serverless cache and an ODCR
> all bill by the hour. Chapter 10 tears everything down — run it.

## Conventions

- `<ACCOUNT_ID>`, `<AWS_REGION>`, `<YOUR_IP>`, `<ALB_DNS>`, `<VALKEY_ENDPOINT>`, `<ODCR_ID>`
  and `<REDACTED>` are placeholders standing in for account-specific values.
- Values that differ on every run are also generic: `<age>`, `<timestamp>`, `<date>`, `<time>`,
  `<resync>`, `i-<gpu-node-id>` / `i-<system-node-id>` / `i-<general-node-id>`, and pod-name
  hashes shortened to `…` (e.g. `mistral-…`). **Durations that are actual measurements are
  real** — Job `DURATION 13m`, `init engine … 9.16 s`, training `Runtime: 169s`, and every
  benchmark latency are unmodified.
- Directory names (`100-vllm/`, `300-benchmarking/`, …) refer to the repo layout:
  ```
  100-vllm  200-strands-agent  300-benchmarking  400-lmcache  600-finetuning  700-rag  800-ray
  ```
- Command blocks are copied from the run. Output blocks are real output, trimmed.
- Each chapter opens with an **"In plain terms"** blockquote: the intuition and vocabulary for
  that chapter, with *Background* links into [CONCEPTS.md](CONCEPTS.md). Skip it freely if the
  topic is already familiar — no command depends on it.

## Credits

The lab environment and manifests originate from the AWS **GenAI on EKS** workshop
(AWS Workshop Studio). This tutorial is my own write-up of an actual execution: the
sequencing, the measurements, the failures, and the fixes. Screenshots embedded here are
my own captures of the running applications (Grafana, Gradio); the workshop's own pages
are not reproduced.
