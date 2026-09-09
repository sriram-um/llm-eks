# 02 — Serve the model with vLLM, straight from S3

Goal: run `Ministral-3-8B-Instruct-2512` on the L40S with an OpenAI-compatible API, without
ever baking 10.4 GB of weights into a container image or copying them onto a volume first.

> **In plain terms.** An *inference server* is a web server with a model inside it. It accepts
> HTTP requests containing text, runs the model, and streams text back. **vLLM** is one, and it
> deliberately speaks the same request/response dialect as OpenAI's API — which is why, later in
> this tutorial, three completely different applications (a chat UI, an agent framework, a Gradio
> app) all talk to a self-hosted model without a single line of custom client code.
>
> Two things in this chapter are worth slowing down for:
>
> **1. The weights never touch a disk.** The obvious way to serve a model is: download 10.4 GB,
> write it to a volume, load it into GPU memory. That's three copies and a lot of waiting.
> `--load-format=runai_streamer` with `--model=s3://…` skips the middle step entirely — it opens
> many parallel ranged reads against S3 and streams bytes into VRAM. **No volume, no PVC, no
> init container.** The container image stays small, and updating the model is an S3 upload.
>
> **2. The startup log is a capacity plan, not noise.** When vLLM boots, it measures the card and
> prints exactly how many concurrent users it can serve. That's the most valuable output in this
> whole tutorial and everyone scrolls past it:
>
> ```
> Available KV cache memory: 27.91 GiB      ← 48 GB card − ~17.8 GB of weights
> GPU KV cache size: 215,152 tokens         ← 27.91 GiB ÷ ~136 KB per token
> Maximum concurrency for 8,192 tokens per request: 26.26x   ← 215,152 ÷ 8,192
> ```
>
> The **KV cache** is where the model keeps its working memory about the conversation so far, and
> it grows with every token of every in-flight request. It, not compute, is what limits how many
> users fit on a card. Chapter 05 measures reality against that `26.26x` and finds it honest.
>
> *Background:* [CONCEPTS Part 1](CONCEPTS.md#part-1--what-a-model-actually-does-when-you-send-it-a-prompt)
> — tokens, prefill vs decode, and why the KV cache is the capacity limit.

## The one flag that matters

```bash
envsubst < 100-vllm/vllm-deployment.yml > /tmp/vllm-temp.yml && mv /tmp/vllm-temp.yml 100-vllm/vllm-deployment.yml
cat 100-vllm/vllm-deployment.yml | grep -A 2 "model=s3://"
kubectl apply -f 100-vllm/vllm-deployment.yml
```

```
            - '--model=s3://genai-models-<ACCOUNT_ID>/Ministral-3-8B-Instruct-2512/'
            - '--served-model-name=ministral'
            - '--load-format=runai_streamer'
service/vllm-serve-svc created
deployment.apps/mistral created
```

Three arguments carry the whole design:

- **`--model=s3://…`** — vLLM takes an S3 URI directly. No `aws s3 cp`, no init container, no
  PVC, no 10 GB layer in your image.
- **`--load-format=runai_streamer`** — the [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)
  reads safetensors from object storage with many concurrent ranged GETs and streams tensors
  into GPU memory *while* download is still in flight. Compare to the naive path (download
  10.4 GB to disk, then read it back into VRAM), which serializes two full passes over the
  data.
- **`--served-model-name=ministral`** — the API-facing alias. Clients say `"model": "ministral"`
  and never learn the S3 path. This becomes important in Chapter 07 when a LoRA adapter is
  registered as a second "model" with `ministral` as its parent.

There's a real architectural consequence here: **the pod is stateless with respect to model
weights**. Scaling to N replicas costs N × (S3 read), not N × (provision volume, copy, mount).
Rolling out a new model version is an S3 prefix change.

## Wait for the GPU node and the pod

```bash
kubectl wait node --for=condition=Ready -l karpenter.sh/nodepool=gpu
```

```
node/i-<gpu-node-id> condition met
```

```bash
kubectl wait --for=condition=Ready pod -l model=mistral --timeout=300s \
  && kubectl logs -l model=mistral --tail=50
```

```
pod/mistral-… condition met
...
INFO [gpu_model_runner.py:5920] Encoder cache will be initialized with a budget of 8192 tokens,
     and profiled with 2 image items of the maximum feature size.
INFO [gpu_worker.py:462]     Available KV cache memory: 27.91 GiB
INFO [kv_cache_utils.py:1710] GPU KV cache size: 215,152 tokens
INFO [kv_cache_utils.py:1711] Maximum concurrency for 8,192 tokens per request: 26.26x
INFO [core.py:306]            init engine (profile, create kv cache, warmup model) took 9.16 s
WARNING [vllm.py:942] Enforce eager set, disabling torch.compile and CUDAGraphs.
INFO [api_server.py:617] Starting vLLM server on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Read the startup log as a capacity plan

| Line | What it tells you |
|---|---|
| `Available KV cache memory: 27.91 GiB` | After ~17.8 GB of BF16 weights, vLLM claimed the rest of the 48 GB card for KV cache |
| `GPU KV cache size: 215,152 tokens` | Total tokens (prompt + generated) resident across all in-flight requests |
| `Maximum concurrency for 8,192 tokens per request: 26.26x` | ~26 concurrent max-length requests before the scheduler starts queueing |

That last number is your concurrency budget, printed for free at boot. Chapter 05 measures
what actually happens when you exceed it — and the numbers line up.

Two warnings worth understanding rather than ignoring:

- `Enforce eager set, disabling torch.compile and CUDAGraphs` — this deployment runs with
  `--enforce-eager`, which trades ~10–20 % steady-state throughput for a much faster, more
  predictable startup and lower memory during profiling. Fine for a workshop, worth
  revisiting for production. See [GOTCHAS](GOTCHAS.md#vllm-runs-with---enforce-eager).
- `Encoder cache … profiled with 2 image items` — Ministral-3 is multimodal, so vLLM reserves
  encoder budget even for text-only serving.

## Test the OpenAI-compatible API

```bash
kubectl port-forward svc/vllm-serve-svc 8000:8000
```

In a second terminal:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ministral","prompt":"San Francisco is a city that has","max_tokens":7,"temperature":0}' | jq
```

The full route table from the startup log shows what else you get for free — this is worth
scanning once, because it determines how little glue code your applications need:

```
/v1/completions          /v1/chat/completions      /v1/chat/completions/batch
/v1/models               /v1/responses             /v1/messages
/v1/messages/count_tokens                          /tokenize  /detokenize
/health  /ping  /metrics  /load  /version          /invocations
```

- `/v1/chat/completions` + `/v1/models` → Open WebUI, LangChain, the OpenAI SDK, and the
  Strands agent in Chapter 04 all work unmodified.
- `/v1/messages` → the Anthropic Messages API shape, also served.
- **`/metrics`** → Prometheus text format. This is what Chapter 03 scrapes; no exporter
  sidecar needed.
- `/health` → what the Deployment's readiness probe uses, and what `kubectl wait` is really
  waiting on.

## Put a chat UI in front of it

`100-vllm/openwebui.yml` is three objects: a Deployment, a ClusterIP Service, and an ALB
Ingress.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: open-webui
  namespace: default
  labels:
    app: open-webui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: open-webui
  template:
    metadata:
      labels:
        app: open-webui
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: "m5.xlarge"     # keep the UI off the GPU node
      containers:
      - name: open-webui
        image: ghcr.io/open-webui/open-webui:v0.11.0
        ports:
        - containerPort: 8080
        resources:
          requests: {cpu: "500m", memory: "500Mi"}
          limits:   {cpu: "1000m", memory: "1Gi"}
        env:
        - name: OPENAI_API_BASE_URLS
          value: "http://vllm-serve-svc:8000/v1"          # ← the entire integration
        - name: OPENAI_API_KEY
          value: "dummy"                                  # vLLM has no auth here
        - name: WEBUI_AUTH
          value: "False"
        - name: ENABLE_OLLAMA_API
          value: "False"
        - name: ENABLE_EVALUATION_ARENA_MODELS
          value: "False"
        volumeMounts:
        - name: webui-volume
          mountPath: /app/backend/data
      volumes:
      - name: webui-volume
        emptyDir: {}                                      # chats are ephemeral
---
apiVersion: v1
kind: Service
metadata:
  name: open-webui
  namespace: default
  labels: {app: open-webui}
spec:
  type: ClusterIP
  selector: {app: open-webui}
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: open-webui-ingress
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '10'
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: '9'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '10'
    alb.ingress.kubernetes.io/success-codes: '200-302'
    alb.ingress.kubernetes.io/load-balancer-name: open-webui-ingress
    alb.ingress.kubernetes.io/inbound-cidrs: 0.0.0.0/0
  labels: {app: open-webui}
spec:
  ingressClassName: alb
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: open-webui
            port: {number: 80}
```

Notes on the parts people get wrong:

- **`OPENAI_API_BASE_URLS=http://vllm-serve-svc:8000/v1`** — that's the whole integration.
  Because vLLM speaks OpenAI, the UI needs no adapter. It calls `/v1/models`, discovers
  `ministral`, and populates the model picker.
- **`success-codes: '200-302'`** — Open WebUI redirects on `/`. With the default `200` the
  ALB target group flaps unhealthy forever.
- **`inbound-cidrs: 0.0.0.0/0`** is overridden by the `IngressClassParams` patch from
  Chapter 00. Belt and braces: verify the resulting security group, don't trust the
  annotation.
- **`emptyDir`** for `/app/backend/data` means chat history dies with the pod. Correct for a
  lab; swap for a PVC if you want it to persist.
- **`nodeSelector: m5.xlarge`** keeps the UI on a CPU node. Never let a web frontend consume
  GPU-node CPU.

```bash
kubectl apply -f 100-vllm/openwebui.yml
kubectl wait --for=condition=ready pod -l app=open-webui --timeout=300s

export OPENWEBUI_URL=$(kubectl get ingress open-webui-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# An ALB's DNS name exists before the ALB is usable — wait for it properly
aws elbv2 wait load-balancer-available --load-balancer-arns \
  $(aws elbv2 describe-load-balancers \
      --query 'LoadBalancers[?DNSName==`'"$OPENWEBUI_URL"'`].LoadBalancerArn' --output text)

echo "Open WebUI is ready at: http://${OPENWEBUI_URL}"
```

```
pod/open-webui-… condition met
Open WebUI is ready and available at: http://<ALB_DNS>
```

That `aws elbv2 wait load-balancer-available` line is the difference between a working demo
and 60 seconds of `503` while you refresh and doubt yourself. `kubectl get ingress` returns
a hostname as soon as the controller *creates* the load balancer; provisioning and target
registration take longer.

Next: [03 — Observability](03-observability.md)
