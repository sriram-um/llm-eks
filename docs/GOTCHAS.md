# GOTCHAS — everything that went wrong, and what to do about it

Every entry below is something that actually happened during the run this tutorial documents.
Nothing here is hypothetical. Several of these fail *silently* or with a misleading message,
which is why they're collected in one place.

| # | Symptom | Severity |
|---|---|---|
| [1](#model-download-job-shows-initerror) | `model-download` pod shows `Init:Error` | Cosmetic — the Job succeeded |
| [2](#kubectl-wait-before-helm-install) | `error: no matching resources found` | Sequencing mistake |
| [3](#inference-perf-cannot-load-the-ministral-tokenizer) | Benchmark can't load the model's tokenizer | Needs a substitute tokenizer |
| [4](#lmcache-pythonhashseed) | Shared KV cache silently never hits | **Silent, production-breaking** |
| [5](#lora-training-needs-remove_columnsmessages) | SFTTrainer re-applies the chat template | Corrupts training data |
| [6](#flash-attn-version-outside-supported-range) | `Found flash-attn 2.8.3` warning | Benign |
| [7](#vllm-runs-with---enforce-eager) | vLLM logs a performance warning at startup | Real throughput cost |
| [8](#helm-install-is-not-idempotent) | `cannot re-use a name that is still in use` | Annoying, harmless |
| [9](#rayservice-never-reaches-running-within-10-minutes) | `kubectl wait` on RayService times out | Timeout too short |
| [10](#open-webui-points-at-a-service-that-doesnt-exist) | Chat UI loads but lists no models | **Silent** |
| [11](#per-ingress-inbound-cidrs-overrides-the-cluster-wide-ip-lock) | ALB open to `0.0.0.0/0` despite the lock | **Security** |
| [12](#servicemonitor--podmonitor-without-the-release-label) | Metrics never appear in Grafana | **Silent** |
| [13](#odcr-cleanup-derives-the-wrong-region) | Capacity reservation never cancelled | **Silent, costs money** |
| [14](#terraform-destroy-doesnt-exist-in-workshop-studio) | `terraform: command not found` | Expected |

---

## `model-download` Job shows `Init:Error`

**Symptom**

```
NAME               READY   STATUS       RESTARTS   AGE
model-download-…   0/1     Init:Error   0          13m
model-download-…   0/1     Completed    0          11m
```

One pod failed. It looks like the model download broke.

**What's actually happening**

It didn't. A Kubernetes `Job` retries on failure by creating a *new pod* and leaves the failed
one behind for inspection. The first attempt hit a transient error (in this run, during the S3
transfer of a 10.4 GB file); the retry completed.

**Check the Job, not the pods**

```bash
kubectl get job model-download
```

```
NAME             STATUS     COMPLETIONS   DURATION   AGE
model-download   Complete   1/1           13m        <age>
```

`Complete 1/1` is the authoritative answer. Failed pods from a succeeded Job are diagnostic
artifacts, not errors.

**Rule of thumb:** for Jobs, `kubectl get job` is the source of truth. For a real failure you'd
see the Job stuck at `0/1` with `BackoffLimitExceeded`.

---

## `kubectl wait` before `helm install`

**Symptom**

```
error: no matching resources found
```

**Cause**

Running the wait command before the resource exists:

```bash
# Wrong order
kubectl wait --for=condition=complete job -l benchmark.scenario=saturation --timeout=600s
helm install saturation 300-benchmarking/benchmark-charts -f /tmp/saturation-values.yaml
```

`kubectl wait` with a **label selector** fails immediately when zero objects match — it does
not poll waiting for one to appear. (`kubectl wait` on a *named* resource behaves the same way.)

**Fix**

Install first, then wait:

```bash
helm install saturation 300-benchmarking/benchmark-charts -f /tmp/saturation-values.yaml
kubectl wait --for=condition=complete job -l benchmark.scenario=saturation --timeout=600s
```

If you're scripting something that races, poll for existence first:

```bash
until kubectl get job -l benchmark.scenario=saturation 2>/dev/null | grep -q saturation; do
  sleep 2
done
```

Relevant chapter: [05 — Benchmarking](05-benchmarking.md#run-the-saturation-scenario).

---

## `inference-perf` cannot load the Ministral tokenizer

**Symptom**

`inference-perf` fails to start when pointed at the served model's own tokenizer.

**Cause**

`Ministral-3-8B-Instruct-2512`'s `tokenizer_config.json` references `TokenizersBackend`, a
`mistral-common` class that HuggingFace's `AutoTokenizer` cannot resolve. The benchmark needs a
local tokenizer to generate synthetic prompts of an exact token length and to count output
tokens.

**Fix — substitute a compatible tokenizer with the same vocabulary**

```yaml
  target:
    modelName: ministral                                  # what vLLM serves
    tokenizerPath: mistralai/Mistral-Nemo-Instruct-2407   # what the harness tokenizes with
```

Mistral-Nemo uses the same **131k Tekken** vocabulary with a standard HuggingFace tokenizer
class, so token counts match and the measurements stay valid.

**Why this is safe, and when it wouldn't be:** the substitution is only legitimate because the
vocabularies are identical. Pairing a model with an unrelated tokenizer would silently
misreport every prompt length and tokens/sec figure. Verify vocab equivalence before doing
this — don't just pick a tokenizer that loads.

Also required, for the same reason:

```yaml
  dependencies:
    packages: [sentencepiece, protobuf]
```

Relevant chapter: [05 — Benchmarking](05-benchmarking.md#the-benchmark-chart).

---

## LMCache `PYTHONHASHSEED`

**This is the most dangerous entry in this file**, because everything appears to work.

**Symptom**

```
LMCache WARNING: Could not load 'builtin' from vLLM. Using builtin hash. This may cause
                 inconsistencies in distributed caching.
LMCache WARNING: Using builtin hash without PYTHONHASHSEED set. For production environments
                 (non-testing scenarios), you MUST set PYTHONHASHSEED to ensure consistent
                 hashing across processes. Example: export PYTHONHASHSEED=0
```

**Why it matters**

LMCache keys cache entries by a hash of the token sequence. **Python randomizes `hash()` per
process** unless `PYTHONHASHSEED` is set. So:

- Pod A stores KV tensors under key `H_A(tokens)`.
- Pod B looks up the identical prompt and computes `H_B(tokens)` ≠ `H_A(tokens)`.
- Every cross-pod lookup misses. Your Valkey L2 cache fills with entries nothing can ever read.
- The same thing happens to a single pod across a restart.

No error, no failed request, no alert. You pay for ElastiCache and get 0 % hit rate. The single-pod
tests in [Chapter 06](06-kv-cache-offloading.md) still showed 14–20× speedups precisely because
one process is self-consistent — which is exactly what makes this easy to miss in testing.

**Fix**

```yaml
        env:
        - name: PYTHONHASHSEED
          value: "0"
```

Set it on **every** pod that shares a cache, and treat it as part of the cache's identity: if
one replica has a different value, it's effectively using a different cache.

**Verify** the cache is actually being read, don't assume:

```bash
kubectl run vcheck --rm -i --restart=Never --image=redis:7-alpine -- \
  redis-cli --tls -h $VALKEY_ENDPOINT -p 6379 dbsize

kubectl logs <pod> | grep -E "Retrieved|hit tokens"
```

`Retrieved 4096 out of 4096 required tokens` on a pod that never saw the prompt is proof. A
growing `dbsize` alone is not — that only proves writes.

Relevant chapter: [06 — KV cache offloading](06-kv-cache-offloading.md#part-1--l1-offload-to-cpu-ram).

---

## LoRA training needs `remove_columns(["messages"])`

**Symptom**

`SFTTrainer` applies the chat template a second time, producing doubly-templated training
examples (nested `[INST]` blocks and duplicated special tokens). Training "succeeds" and the
adapter learns the wrong thing.

**Cause**

`format_chat` renders `messages` into a `text` field, but the original `messages` column stays
in the dataset. TRL inspects the columns, sees a conversational field, and treats the dataset
as conversational — templating it again.

**Fix**

```bash
sed -i 's/dataset = dataset.map(format_chat)/dataset = dataset.map(format_chat)\ndataset = dataset.remove_columns(["messages"])/' \
  600-finetuning/train_lora.py
```

i.e.

```python
dataset = dataset.map(format_chat)
dataset = dataset.remove_columns(["messages"])   # ← leave only `text`
```

**How to catch this class of bug generally:** print one fully-tokenized training example and
read it before launching a run. Doubled special tokens are obvious on inspection and nearly
invisible in the loss curve.

Relevant chapter: [07 — LoRA fine-tuning](07-lora-fine-tuning.md#patch-the-training-script).

---

## flash-attn version outside supported range

**Symptom**

```
Supported flash-attn versions are >= 2.1.1, <= 2.7.4.post1. Found flash-attn 2.8.3.
```

**Verdict: benign here.** Training completed normally — 250 steps, final loss 0.1811, adapter
written to S3.

The warning comes from `transformers` guarding against a version it hasn't validated. It may
fall back to a different attention implementation, which affects speed rather than correctness.

If you want determinism, pin it:

```bash
pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

For a 169-second training run it isn't worth the build time. For anything reproducible, pin
every version rather than letting `pip install` resolve latest at pod startup — which is the
deeper problem with ConfigMap-mounted scripts that install dependencies at runtime.

Relevant chapter: [07 — LoRA fine-tuning](07-lora-fine-tuning.md#watch-the-training-run).

---

## vLLM runs with `--enforce-eager`

**Symptom** (at startup, easy to scroll past)

```
INFO … Cudagraph is disabled under eager mode
WARNING … --enforce-eager is set: torch.compile and CUDA graph capture are disabled.
          This will lower performance.
```

**Cause**

`--enforce-eager` is set on the vLLM command line. It skips `torch.compile` and CUDA graph
capture, so every decode step is dispatched from Python.

**Why the workshop does it:** it cuts startup time substantially (compilation and graph capture
take minutes) and removes a class of compile-time failures. Sensible for a lab where you start
and stop the server repeatedly.

**Why you shouldn't ship it:** CUDA graphs mainly reduce per-step kernel-launch overhead, which
is exactly what dominates single-token decode. Removing it costs real decode throughput.

**Fix**

Drop the flag:

```yaml
        args:
          - --model=s3://…
          - --served-model-name=ministral
          - --load-format=runai_streamer
          # - --enforce-eager        ← remove for production
```

Then re-run [Chapter 05](05-benchmarking.md)'s saturation scenario and compare. **Every number
in this tutorial was measured with `--enforce-eager` on**, so treat them as a floor, not a
ceiling.

Relevant chapter: [02 — Serve with vLLM](02-serve-with-vllm.md#read-the-startup-log-as-a-capacity-plan).

---

## `helm install` is not idempotent

**Symptom**

```
"kuberay" already exists with the same configuration, skipping
Error: INSTALLATION FAILED: cannot re-use a name that is still in use
```

**Cause**

`helm repo add` is idempotent and says so. `helm install` is not. Re-running a copy-pasted setup
block fails on the second command.

**Fix**

```bash
helm upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.1.0
```

Use `helm upgrade --install` in any snippet a reader might run twice — including your own
runbooks.

Relevant chapter: [09 — Ray Serve](09-ray-serve.md#install-the-kuberay-operator).

---

## RayService never reaches `Running` within 10 minutes

**Symptom**

```
$ kubectl wait --for=jsonpath='{.status.serviceStatus}'=Running rayservice/vllm --timeout=600s
error: timed out waiting for the condition on rayservices/vllm
```

Reproduced twice. Nothing was broken — the `vllm` Service and the Ray dashboard on 8265 both
worked.

**Cause**

`serviceStatus: Running` is the *end* of a long chain: head pod scheduled → Karpenter
provisions a GPU node (2–4 min cold) → multi-GB AWS DLC image pull → Ray cluster forms → Serve
app deploys → **vLLM loads 10.4 GB of weights from S3**. Ten minutes isn't enough from cold.

**Fix**

```bash
kubectl wait --for=jsonpath='{.status.serviceStatus}'=Running rayservice/vllm --timeout=1800s
```

And while waiting, watch something informative instead of a single boolean:

```bash
kubectl get pods -l ray.io/cluster -w
kubectl describe rayservice vllm
kubectl get nodeclaims          # is Karpenter still provisioning the GPU node?
```

**The general lesson:** a `kubectl wait` timeout tells you a condition wasn't met, never why.
Before deciding something is broken, look at the pods and the events.

Relevant chapter: [09 — Ray Serve](09-ray-serve.md#reality-check-1-the-readiness-wait-times-out).

---

## Open WebUI points at a Service that doesn't exist

**Symptom**

Open WebUI deploys, the pod is ready, the ALB health check passes, the page loads — and the
model dropdown is empty.

**Cause**

`800-ray/openwebui.yml` carries the Chapter 02 value:

```yaml
        - name: OPENAI_API_BASE_URLS
          value: "http://vllm-serve-svc:8000/v1"
```

But under Ray there is no `vllm-serve-svc`. KubeRay created a Service named **`vllm`**, and
`vllm-serve-svc` was deleted during the Chapter 08 teardown.

**Fix**

```bash
kubectl set env deployment/open-webui OPENAI_API_BASE_URLS="http://vllm:8000/v1"
kubectl rollout status deployment/open-webui
```

**Confirm the target exists before wiring anything to it** — a Service with no endpoints
resolves in DNS and then hangs:

```bash
kubectl get svc
kubectl get endpointslices -l kubernetes.io/service-name=vllm
```

**Design lesson:** the client's contract is the Service *name*, and the operator chooses it. If
you want to swap serving layers transparently, put a stable Service (or Gateway route) in front
of whichever backend is live rather than letting each backend name itself.

Relevant chapter: [09 — Ray Serve](09-ray-serve.md#reality-check-2-open-webui-points-at-the-wrong-service).

---

## Per-Ingress `inbound-cidrs` overrides the cluster-wide IP lock

**Symptom**

[Chapter 00](00-environment-and-cluster.md#lock-the-alb-to-your-ip-before-exposing-anything)
locks every ALB to your IP via `IngressClassParams`:

```bash
kubectl patch ingressclassparams alb --type=merge \
  -p "{\"spec\":{\"inboundCIDRs\":[\"${MY_IP}/32\"]}}"
```

…and then two manifests quietly undo it:

```yaml
    alb.ingress.kubernetes.io/inbound-cidrs: 0.0.0.0/0
```

(`700-rag/rag-gradio-deploy.yml` and `800-ray/openwebui.yml`.)

**Cause**

The per-Ingress annotation takes precedence over the `IngressClassParams` default. The
cluster-wide setting is a default, not a policy.

**Why this is serious in this specific stack**

Both affected apps are unauthenticated LLM frontends — Open WebUI runs with
`WEBUI_AUTH: "False"`. `0.0.0.0/0` on an internet-facing ALB means anyone who finds the DNS name
gets free, unmetered inference on your GPU.

**Fix**

```yaml
    alb.ingress.kubernetes.io/inbound-cidrs: <YOUR_IP>/32
```

Or delete the annotation entirely so the `IngressClassParams` default applies. Then verify at
the security group, not in YAML:

```bash
kubectl get ingress -A -o custom-columns=\
'NS:.metadata.namespace,NAME:.metadata.name,CIDRS:.metadata.annotations.alb\.ingress\.kubernetes\.io/inbound-cidrs'

aws elbv2 describe-load-balancers --query 'LoadBalancers[].[LoadBalancerName,Scheme]' --output table
```

If you need real access control rather than an IP allowlist, terminate auth at the ALB (OIDC /
Cognito) or don't expose it — `kubectl port-forward` is sufficient for everything in this
tutorial.

---

## `ServiceMonitor` / `PodMonitor` without the `release` label

**Symptom**

The monitor object exists, `kubectl get servicemonitor` lists it, the YAML looks correct, and no
metrics ever appear in Prometheus or Grafana. No errors anywhere.

**Cause**

The Prometheus instance created by kube-prometheus-stack has a
`serviceMonitorSelector` / `podMonitorSelector` matching `release: kube-prometheus-stack`.
Objects without that label are invisible to it.

**Fix — required on every monitor in this tutorial**

```yaml
metadata:
  labels:
    release: kube-prometheus-stack
```

Applies to `mistral-monitor`, the DCGM exporter's `serviceMonitor.additionalLabels`,
`ray-head-monitor`, and `ray-workers-monitor`.

**Three related silent failures in the same area:**

- **`port` must be a port *name*, not a number.** The operator resolves names only.
- **`selector.matchLabels` matches the Service (for `ServiceMonitor`), not the pod.** If
  `vllm-serve-svc` lacks `model: mistral`, nothing is discovered.
- **`honorLabels: true` on the DCGM exporter.** Without it Prometheus overwrites DCGM's `pod` /
  `container` labels with the exporter's own identity, and you lose per-workload GPU
  attribution entirely — the metrics are there but they can't answer "which pod is using the
  GPU?"

**Verify** in Prometheus' own UI rather than in the dashboard:

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
# → Status → Target health. If your job isn't listed, it was never discovered.
```

Relevant chapters: [03 — Observability](03-observability.md#deploy-dcgm-exporter),
[09 — Ray Serve](09-ray-serve.md#observability-ray-metrics-into-the-same-grafana).

---

## ODCR cleanup derives the wrong region

**This one costs money.**

**Symptom**

```
Using AWS Region: genai-workshop

aws: [ERROR]: Could not connect to the endpoint URL: "https://ec2.genai-workshop.amazonaws.com/"
Found ODCR ID: 

aws: [ERROR]: An error occurred (ParamValidation): argument --capacity-reservation-id: expected one argument
```

**Cause**

```bash
AWS_REGION=$(kubectl config get-contexts | grep '*' | awk '{print $2}' | cut -d':' -f4)
```

This assumes the kubectl context name is an EKS cluster ARN
(`arn:aws:eks:us-east-2:123456789012:cluster/name`), where field 4 of a colon-split is the
region. This cluster's context was named `genai-workshop`, so the pipeline returned the context
name. Then `describe-capacity-reservations` failed against a nonexistent endpoint, `ODCR_ID`
came back empty, and the cancel call failed on the missing argument.

**Net effect: the On-Demand Capacity Reservation was never cancelled.** An ODCR bills for its
reserved capacity **whether or not any instance is running in it** — this is the single most
expensive thing to leave behind.

**Fix**

```bash
export AWS_REGION=$(aws configure get region)     # or set it explicitly

aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
  --filters "Name=state,Values=active" \
  --query 'CapacityReservations[].{Id:CapacityReservationId,Type:InstanceType,
            Total:TotalInstanceCount,AZ:AvailabilityZone}' --output table

aws ec2 cancel-capacity-reservation --region "$AWS_REGION" \
  --capacity-reservation-id cr-0123456789abcdef0
```

Two separate lessons:

- **Don't parse identifiers out of context names.** Read the region from AWS config, an
  environment variable, or `aws eks describe-cluster`.
- **Don't blind-index `CapacityReservations[0]`.** It takes whichever active reservation the API
  returns first, which in a shared account may not be yours. List, read, then cancel by
  explicit ID.

And always confirm afterwards:

```bash
aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
  --filters "Name=state,Values=active" --query 'length(CapacityReservations)'
```

Relevant chapter: [10 — Cleanup](10-cleanup.md#4-cancel-the-on-demand-capacity-reservation).

---

## `terraform destroy` doesn't exist in Workshop Studio

**Symptom**

```
bash: cd: sample-genai-on-eks/terraform: No such file or directory
bash: terraform: command not found
```

**Cause**

The cleanup instructions assume you provisioned the cluster yourself. In an AWS Workshop Studio
environment the Terraform state and binary live outside your IDE environment; the workshop
account tears its own infrastructure down when the event ends.

**What to do**

Nothing — this is expected. But it means **you are responsible for everything the workshop
didn't create for you**, in particular:

- the ODCR ([#13](#odcr-cleanup-derives-the-wrong-region))
- the ElastiCache Serverless cache from Chapter 06
- any ALB whose Ingress you didn't delete

Run steps 1–6 of [Chapter 10](10-cleanup.md) and verify with the four `length(...)` queries at
the end. Don't assume the environment expiry covers resources you created by hand.

---

## Cross-cutting patterns

Six of these fourteen fail **silently**: #4 (hash seed), #5 (double templating), #10 (wrong
Service), #11 (open ALB), #12 (missing release label), #13 (uncancelled ODCR). Two more (#1,
#9) look like failures but aren't.

That ratio is the real lesson of this tutorial. A distributed inference stack has many layers
that each degrade quietly:

1. **Verify at the layer you care about, not the layer you configured.** Check Prometheus target
   health, not whether the `ServiceMonitor` exists. Check `Retrieved N of N tokens`, not
   `dbsize`. Check the security group, not the annotation.
2. **A green pod is not a working feature.** Open WebUI was `Ready` with zero models; the LoRA
   Job "succeeded" with doubly-templated data.
3. **`kubectl wait` timeouts are questions, not answers.** Look at pods and events before
   concluding anything is broken.
4. **Cost lives in the resources nothing reconciles.** ALBs, ODCRs, and ElastiCache have no
   controller watching a desired state — if you don't delete them, nothing will.

---

Back to the [index](README.md)
