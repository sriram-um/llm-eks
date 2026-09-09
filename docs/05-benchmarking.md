# 05 — Benchmarking: finding the saturation knee

Goal: stop guessing at capacity. Run `inference-perf` as a Kubernetes Job, sweep the request
rate, and find the exact point where adding load stops buying throughput and starts buying
latency.

The result from this run, up front:

| Stage | Rate | Successes | **TTFT p50** | **Output tok/s** | Req/s |
|---|---|---|---|---|---|
| 0 | 5 req/s | 300 | **81 ms** | 1,156 | 4.43 |
| 1 | 10 req/s | 600 | **242 ms** | 2,103 | 8.17 |
| 2 | 20 req/s | 1,200 | **27.5 s** | 2,253 | 8.89 |

Between stage 1 and stage 2, throughput improved **7 %** and TTFT got **114× worse**. Stage 1
is the knee. That single table is the entire point of this chapter.

> **In plain terms.** Picture a motorway. Add cars and you move more people per hour — up to a
> point. Past that point you don't move more people at all; you just add a traffic jam. Total
> throughput barely changes while every individual journey gets far worse.
>
> Inference servers behave exactly like this, and the reason is queueing, not the model:
>
> ```
> arrivals < capacity  →  short queue, latency ≈ processing time      stable
> arrivals ≈ capacity  →  queue gets twitchy, latency climbs fast     ★ the knee
> arrivals > capacity  →  queue grows for as long as you keep pushing  unstable
> ```
>
> That last row is the crucial one: an overloaded queue does not settle at "a bit longer." It
> grows without bound for as long as the overload lasts. So **throughput degrades gracefully and
> latency degrades catastrophically** — which is precisely the 7 % / 114× split in the table
> above.
>
> Two terms you need to keep separate while reading the results:
>
> - **TTFT (Time To First Token)** — how long before text starts appearing. This includes queue
>   wait, which is why it's the metric that collapses first. Under ~300 ms feels instant; 27.5 s
>   means the user closed the tab.
> - **Throughput (output tok/s)** — total work completed. Looks *fine* in stage 2. It is lying to
>   you.
>
> Hence the practical rule this chapter exists to establish: **your capacity is the knee, not the
> peak-throughput point.** Autoscale on queue depth or TTFT, never on tokens/sec — a saturated
> server and a well-loaded one have nearly identical throughput graphs.
>
> Also watch `p50` / `p99` in the output. `p99` means one request in a hundred was worse than
> that number. A user making 100 requests in a session hits their p99 about once, so tail
> latency is not an edge case — it's the experience.
>
> *Background:* [CONCEPTS Part 2](CONCEPTS.md#part-2--the-four-numbers-and-how-to-read-a-percentile).

## Precondition

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

kubectl get pods -l model=mistral
kubectl get svc vllm-serve-svc
```

```
NAME        READY   STATUS    RESTARTS   AGE
mistral-…   1/1     Running   0          <age>
NAME             TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
vllm-serve-svc   ClusterIP   172.20.230.178   <none>        8000/TCP   <age>
```

## The benchmark chart

`inference-perf` v0.6.1 is packaged as a local Helm chart
(`300-benchmarking/benchmark-charts/`) so that a load test is a versioned, reviewable
artifact rather than a laptop script. Key parts of `values.yaml`:

```yaml
benchmark:
  enabled: true
  image:
    repository: quay.io/inference-perf/inference-perf
    tag: v0.6.1

  serviceAccount:
    create: false
    name: model-storage-sa        # EKS Pod Identity → S3 read+write, no keys in the pod

  scenario: baseline              # baseline | saturation | sweep | production

  job:
    backoffLimit: 2
    ttlSecondsAfterFinished: 3600

  target:
    serverType: vllm
    modelName: ministral                                   # must match /v1/models
    baseUrl: http://vllm-serve-svc.default.svc.cluster.local:8000
    tokenizerPath: mistralai/Mistral-Nemo-Instruct-2407     # ← see note below
    ignoreEos: true                                        # ← critical for valid results

  api:
    type: completion             # synthetic data generation only supports `completion`
    streaming: true              # required to measure TTFT at all

  storage:
    s3:
      bucketName: ""             # set via --set / envsubst
      pathPrefix: "benchmarks"

  dependencies:
    packages: [sentencepiece, protobuf]   # Mistral tokenizer needs these

  resources:
    requests: {cpu: "2", memory: "4Gi"}
    limits:   {cpu: "4", memory: "8Gi"}

  affinity:
    enabled: true
    targetLabels:
      model: mistral             # co-locate the load generator with the target's zone
```

Five design choices that make the difference between a benchmark and a number-generator:

**`ignoreEos: true`** — forces every request to generate exactly `output.mean` tokens instead
of stopping at a natural end-of-sequence. Without it, output length varies with the prompt and
your tokens/sec figure is unreproducible noise.

**`streaming: true`** — TTFT is undefined for a non-streaming request. If you care about
perceived latency, you must stream.

**`affinity.targetLabels: {model: mistral}`** — schedules the load generator in the same zone
as the model server. Cross-AZ round trips add ~1–2 ms, which is invisible at 27 s of queue
time but material when you're measuring an 81 ms TTFT.

**`resources: 2–4 CPU / 4–8 Gi`** — a load generator that is itself CPU-starved will report
your server as slow. Tokenizing thousands of synthetic prompts is real work.

**`tokenizerPath: mistralai/Mistral-Nemo-Instruct-2407`** — a deliberate substitution, not a
mistake. Ministral-3-8B's `tokenizer_config.json` references `TokenizersBackend`, which
`AutoTokenizer` cannot resolve, so `inference-perf` can't load it. Mistral-Nemo shares the
same 131k Tekken vocabulary with a standard HuggingFace tokenizer class, so token counts
match. See [GOTCHAS](GOTCHAS.md#inference-perf-cannot-load-the-ministral-tokenizer).

## Four scenarios, four different questions

```yaml
  scenarios:
    # SCENARIO 1 — "What is the best case?"
    baseline:
      description: "Establish optimal performance with zero contention"
      data:
        input:  {mean: 512, stdDev: 0, min: 512, max: 512}   # fixed size = zero variance
        output: {mean: 128, stdDev: 0, min: 128, max: 128}
      load:
        type: constant
        numWorkers: 4
        stages:
          - {rate: 1, duration: 300}                          # 1 req/s — no queueing

    # SCENARIO 2 — "Where does it break?"
    saturation:
      description: "Determine maximum sustainable throughput"
      data:
        input:  {mean: 512, stdDev: 128, min: 128, max: 2048}
        output: {mean: 256, stdDev: 64,  min: 32,  max: 512}
      load:
        type: constant
        numWorkers: 8
        stages:
          - {rate: 5,  duration: 180}
          - {rate: 10, duration: 180}
          - {rate: 20, duration: 180}
          - {rate: 40, duration: 180}

    # SCENARIO 3 — "Find the knee for me"
    sweep:
      description: "Automated capacity discovery with intelligent staging"
      data:
        input:  {mean: 512, stdDev: 128, min: 128, max: 2048}
        output: {mean: 256, stdDev: 64,  min: 32,  max: 512}
      load:
        type: constant
        numWorkers: 8
        stages: []              # auto-generated
        sweep:
          type: geometric       # geometric rate spacing
          numRequests: 2000
          timeout: 60
          numStages: 5
          stageDuration: 180
          saturationPercentile: 95

    # SCENARIO 4 — "What does real traffic look like?"
    production:
      description: "Realistic traffic with variable sizes and bursty arrivals"
      data:
        input:  {mean: 1024, stdDev: 512, min: 128, max: 4096}
        output: {mean: 512,  stdDev: 256, min: 50,  max: 2048}
      load:
        type: poisson           # bursty arrivals, not a metronome
        numWorkers: 8
        stages:
          - {rate: 15, duration: 600}
```

The progression is the methodology:

| Scenario | Distribution | Arrival | Answers |
|---|---|---|---|
| baseline | fixed 512→128 | constant 1/s | Floor latency with zero contention. Your SLO's best case. |
| saturation | 512±128 → 256±64 | constant, 4 stages | Manual rate ladder. Where the knee is. |
| sweep | same | geometric, auto | Machine-found knee at p95. Repeatable in CI. |
| production | 1024±512 → 512±256 | **Poisson** | Bursts. A Poisson process at mean 15/s hits 30/s regularly; a constant 15/s never does. |

Most benchmark reports are the `baseline` scenario with a big number attached. The
`production` scenario is the one that resembles a real service, and it's the one where
queueing theory bites: **the same mean rate with bursty arrivals produces materially worse
tail latency** than a metronome.

## Run the saturation scenario

```bash
envsubst < 300-benchmarking/saturation-values.yaml > /tmp/saturation-values.yaml

helm install saturation 300-benchmarking/benchmark-charts \
  -n default \
  -f /tmp/saturation-values.yaml \
  --set benchmark.serviceAccount.create=false
```

```
NAME: saturation
NAMESPACE: default
STATUS: deployed
NOTES:
Benchmark job deployed: SATURATION
Target: http://vllm-serve-svc.default.svc.cluster.local:8000
Model: ministral
To monitor:
  kubectl logs -n default -l benchmark.scenario=saturation -f
Results: s3://genai-models-<ACCOUNT_ID>/benchmarks
```

```bash
kubectl wait --for=condition=complete job -l benchmark.scenario=saturation \
  -n default --timeout=600s
```

```
job.batch/saturation-benchmark-charts-saturation condition met
```

> Run `helm install` **before** `kubectl wait`. Waiting first gives you
> `error: no matching resources found` — see [GOTCHAS](GOTCHAS.md#kubectl-wait-before-helm-install).

While it runs, watch the Grafana engine-metrics dashboard from
[Chapter 03](03-observability.md#the-vllm-engine-metrics-dashboard). That screenshot was taken
during this exact run.

## Read the results out of S3

Results are written to S3 by the Job (via the `model-storage-sa` Pod Identity), so they
outlive the pod:

```bash
SATURATION_KEY=$(aws s3 ls "s3://${S3_BUCKET_NAME}/benchmarks/saturation/vllm/ministral/" --recursive \
  | grep 'stage_0_lifecycle_metrics.json' \
  | sort -k1,2 | tail -n1 | awk '{print $4}')
SATURATION_BEFORE="s3://${S3_BUCKET_NAME}/$(dirname "${SATURATION_KEY}")"
echo "$SATURATION_BEFORE" > ~/saturation_before_path.txt
echo "Saved before path: $SATURATION_BEFORE"

for stage in 0 1 2; do
  aws s3 cp "${SATURATION_BEFORE}/stage_${stage}_lifecycle_metrics.json" - 2>/dev/null \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
s = d['successes']
ttft = s['latency']['time_to_first_token']
tps  = s['throughput']
ttft_ms = ttft['median']*1000
ttft_str = f'{ttft_ms/1000:.1f}s' if ttft_ms > 1000 else f'{ttft_ms:.0f}ms'
print(f'Stage ${stage}: {s[\"count\"]:>5} ok | TTFT p50: {ttft_str:>8} | Output TPS: {tps[\"output_tokens_per_sec\"]:>8.1f} | Req/s: {tps[\"requests_per_sec\"]:.2f}')
"
done
```

```
Saved before path: s3://genai-models-<ACCOUNT_ID>/benchmarks/saturation/vllm/ministral/YLeqFTRZoumohdpZrjQQ
Stage 0:   300 ok | TTFT p50:     81ms | Output TPS:   1156.1 | Req/s: 4.43
Stage 1:   600 ok | TTFT p50:    242ms | Output TPS:   2103.4 | Req/s: 8.17
Stage 2:  1200 ok | TTFT p50:    27.5s | Output TPS:   2253.4 | Req/s: 8.89
```

Persisting results to S3 rather than reading them from `kubectl logs` is the right call:
`ttlSecondsAfterFinished: 3600` deletes the Job after an hour, and you want to compare today's
run against a run from three weeks ago.

## The analysis

```
Rate       Achieved   Output tok/s   TTFT p50     Verdict
 5 req/s   4.43       1,156          81 ms        Underutilized — headroom available
10 req/s   8.17       2,103          242 ms       ★ THE KNEE — near-max throughput, usable latency
20 req/s   8.89       2,253          27,500 ms    Saturated — queueing, not serving
40 req/s   —          —              —            (stage never produced usable results)
```

**Stage 0 → 1:** requested rate doubled, achieved rate doubled (4.43 → 8.17), throughput
nearly doubled (1,156 → 2,103 tok/s). Latency rose 3× but 242 ms is still fine for a
streaming UI. The GPU was underutilized at stage 0.

**Stage 1 → 2:** requested rate doubled again but achieved rate moved only 8.17 → 8.89 (+9 %)
and throughput 2,103 → 2,253 (+7 %). The server was already at capacity. The extra load went
straight into the queue: **TTFT p50 242 ms → 27.5 s, a 114× regression**.

The mechanism is visible in the [Chapter 03 dashboard](03-observability.md#the-vllm-engine-metrics-dashboard):
`Num Running` plateaus while `Num Waiting` spikes to ~440. vLLM's continuous batcher was full
— exactly the `Maximum concurrency for 8,192 tokens per request: 26.26x` that vLLM printed at
startup. Requests beyond that ceiling wait.

Notice also that `Prefix cache hits %` sat at **0 %** throughout. Synthetic benchmark prompts
share no common prefix, so every request paid full prefill cost. Real traffic — shared system
prompts, multi-turn chat, RAG with a stable instruction block — does not look like that.
Exploiting it is [Chapter 06](06-kv-cache-offloading.md).

### What to take away

1. **Peak throughput and acceptable latency are different operating points.** If you
   provision for max tokens/sec, you ship a 27-second TTFT.
2. **Measure achieved rate, not offered rate.** Stage 2 "ran at 20 req/s" and delivered 8.89.
3. **The knee is your autoscaling target.** Scale out replicas at ~8 req/s per pod rather than
   pushing one pod to its throughput ceiling.
4. **Latency degrades non-linearly, throughput degrades gracefully.** Alert on TTFT p95 and
   queue time, not on tokens/sec.

## Tear down

```bash
helm uninstall baseline -n default 2>/dev/null || true
helm uninstall saturation -n default 2>/dev/null || true
kubectl delete deployment mistral 2>/dev/null || true
kubectl delete service vllm-serve-svc 2>/dev/null || true
```

```
release "baseline" uninstalled
release "saturation" uninstalled
deployment.apps "mistral" deleted from default namespace
```

Next: [06 — KV cache offloading](06-kv-cache-offloading.md)
