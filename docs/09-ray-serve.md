# 09 — Ray Serve on EKS with KubeRay

Chapters 02–08 ran vLLM as a plain `Deployment`. That's the right default. This chapter swaps
the serving layer for **Ray Serve**, managed by the **KubeRay operator**, to show the
alternative and — honestly — where it costs you.

Read this chapter as a comparison, not a recommendation. Two things did not go smoothly, and
both are documented below rather than edited out.

> **In plain terms.** **Ray** is a framework for spreading Python work across many machines, and
> **Ray Serve** is its model-serving layer. It solves problems a plain Kubernetes `Deployment`
> can't: a model too large for a single machine (split across GPUs on *different* nodes), or a
> pipeline of several models that need to scale independently.
>
> Neither of those applies to this tutorial. One 8.9 B model fits comfortably on one L40S. So
> this chapter is a deliberate look at what the extra machinery *costs* when you don't need it:
> **eight Service ports instead of one, an extra operator to run, more places to fail, and a
> readiness wait that timed out at ten minutes — twice.**
>
> That isn't a knock on Ray. It's the ordinary trade of every distributed system: you get a
> distributed runtime, and you also inherit a distributed runtime. Worth knowing before you adopt
> one, which is why the two things that went wrong are left in.
>
> There's a genuinely good result hiding in here too. The observability stack from
> [Chapter 03](03-observability.md) keeps working across the swap, and the total cost of
> monitoring an entirely different serving framework is **two `PodMonitor` files**. That's the
> payoff for treating monitoring as Kubernetes objects rather than per-application config.

## Install the KubeRay operator

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/

helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0

kubectl wait pods --for=jsonpath='{.status.phase}'=Running \
  -l app.kubernetes.io/name=kuberay-operator --timeout=300s
```

```
"kuberay" has been added to your repositories
NAME: kuberay-operator
NAMESPACE: default
STATUS: deployed
REVISION: 1
pod/kuberay-operator-… condition met
```

> Re-running this block gives
> `Error: INSTALLATION FAILED: cannot re-use a name that is still in use`. `helm repo add` is
> idempotent; `helm install` is not. Use `helm upgrade --install` if you're going to run a
> snippet twice. See [GOTCHAS](GOTCHAS.md#helm-install-is-not-idempotent).

## Deploy the RayService

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

envsubst < 800-ray/ray-vllm-service.yaml | kubectl apply -f -
```

```
rayservice.ray.io/vllm created
```

Same model, same S3 bucket, same GPU — a different orchestration layer. The `RayService` CR
tells KubeRay to build a Ray cluster (head + workers) and deploy a Serve application onto it.
This manifest uses an **AWS Deep Learning Container** image rather than the upstream vLLM
image, because the Ray + vLLM + CUDA version matrix is genuinely painful to assemble yourself.

## Reality check #1: the readiness wait times out

```bash
kubectl wait --for=jsonpath='{.status.serviceStatus}'=Running rayservice/vllm --timeout=600s
```

```
error: timed out waiting for the condition on rayservices/vllm
```

Ten minutes was not enough. Reproduced on a second attempt with the same result. Nothing was
broken — the Service and dashboard both came up — but `serviceStatus: Running` requires the
whole chain to settle:

1. KubeRay creates the head pod → Karpenter provisions a node
2. Karpenter provisions a **GPU** node for the worker (~2–4 min from cold)
3. The DLC image pulls (multiple GB)
4. Ray head and worker join the cluster
5. Ray Serve deploys the application
6. **vLLM loads 10.4 GB of weights from S3**
7. Only then does `serviceStatus` flip to `Running`

Compare Chapter 02, where the plain `Deployment` reached ready and vLLM logged
`init engine … took 9.16 s`. The engine start is the same; the orchestration around it is
what got slower.

Use a longer timeout (`--timeout=1800s`) and watch the actual state rather than a single
condition:

```bash
kubectl get rayservice vllm -o yaml | grep -A 10 status:
kubectl get pods -l ray.io/cluster
kubectl describe rayservice vllm
```

## The Ray dashboard

```bash
kubectl port-forward svc/vllm 8265:8265
```

```
Forwarding from 127.0.0.1:8265 -> 8265
Forwarding from [::1]:8265 -> 8265
```

The dashboard at `http://localhost:8265` is the strongest argument for Ray here: Serve
deployment status, replica counts, per-actor logs, and the Ray cluster's own resource
accounting in one place — none of which a bare Deployment gives you.

```bash
kubectl get svc
```

```
NAME               TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                                                                       AGE
kuberay-operator   ClusterIP   172.20.75.168    <none>        8080/TCP                                                                      <age>
kubernetes         ClusterIP   172.20.0.1       <none>        443/TCP                                                                       <age>
vllm               ClusterIP   172.20.180.173   <none>        44227/TCP,6379/TCP,8265/TCP,10001/TCP,8000/TCP,52365/TCP,8080/TCP,44217/TCP   <age>
```

**Eight ports on one Service.** That single line is the complexity delta versus Chapter 02's
`vllm-serve-svc   ClusterIP   …   8000/TCP`:

| Port | Purpose |
|---|---|
| 8000 | Ray Serve HTTP — the OpenAI-compatible endpoint you actually call |
| 8265 | Ray dashboard |
| 6379 | GCS (Global Control Store) — Ray's cluster metadata |
| 10001 | Ray Client |
| 52365 | Dashboard agent |
| 8080 | Metrics |
| 44227, 44217 | Ephemeral worker ports |

Everything except 8000 exists to run Ray. That's the trade: you get a distributed runtime, and
you inherit a distributed runtime.

## Reality check #2: Open WebUI points at the wrong Service

`800-ray/openwebui.yml` is byte-for-byte the Chapter 02 manifest:

```yaml
        env:
        - name: OPENAI_API_BASE_URLS
          value: "http://vllm-serve-svc:8000/v1"    # ← the DEPLOYMENT's Service name
```

```bash
kubectl apply -f 800-ray/openwebui.yml

kubectl wait --for=condition=ready pod -l app=open-webui --timeout=300s

export OPENWEBUI_URL=$(kubectl get ingress open-webui-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
aws elbv2 wait load-balancer-available --load-balancer-arns \
  $(aws elbv2 describe-load-balancers \
      --query 'LoadBalancers[?DNSName==`'"$OPENWEBUI_URL"'`].LoadBalancerArn' --output text)
echo "Open WebUI is ready and available at: http://${OPENWEBUI_URL}"
```

```
deployment.apps/open-webui created
service/open-webui created
ingress.networking.k8s.io/open-webui-ingress created
pod/open-webui-… condition met
Open WebUI is ready and available at: http://<ALB_DNS>
```

But `kubectl get svc` above shows no `vllm-serve-svc` — it was deleted at the end of
[Chapter 08](08-rag-s3-vectors.md#clean-up), and KubeRay created a Service named **`vllm`**.
Open WebUI will come up, pass its health check, and list no models.

Fix it before you open the URL:

```bash
kubectl set env deployment/open-webui OPENAI_API_BASE_URLS="http://vllm:8000/v1"
kubectl rollout status deployment/open-webui
```

This is the generic hazard when you change serving layers: **the client contract is the Service
name, and the operator picks the Service name.** If you want the swap to be transparent to
every consumer, put a stable Service in front of whichever backend is live, rather than
letting each backend define its own name.

Note also `inbound-cidrs: 0.0.0.0/0` on this Ingress, which overrides the cluster-wide IP lock
from [Chapter 00](00-environment-and-cluster.md#lock-the-alb-to-your-ip-before-exposing-anything) — the same caveat as the
Gradio app. `WEBUI_AUTH: "False"` means anyone who reaches it gets an unauthenticated chat UI.

## Observability: Ray metrics into the same Grafana

Ray publishes its own metrics from head and worker pods. The head exposes **three** distinct
metrics ports, hence three `podMetricsEndpoints`.

`800-ray/ray-podmonitor.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  labels:
    release: kube-prometheus-stack   # Required for Prometheus operator discovery
  name: ray-head-monitor
  namespace: monitoring
spec:
  jobLabel: ray-head
  namespaceSelector:
    matchNames: [default]
  selector:
    matchLabels:
      ray.io/node-type: head
  podMetricsEndpoints:
    - port: metrics
      relabelings:
        - action: replace
          sourceLabels: [__meta_kubernetes_pod_label_ray_io_cluster]
          targetLabel: ray_io_cluster
    - port: as-metrics        # autoscaler metrics
      relabelings:
        - action: replace
          sourceLabels: [__meta_kubernetes_pod_label_ray_io_cluster]
          targetLabel: ray_io_cluster
    - port: dash-metrics      # dashboard metrics
      relabelings:
        - action: replace
          sourceLabels: [__meta_kubernetes_pod_label_ray_io_cluster]
          targetLabel: ray_io_cluster
---
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: ray-workers-monitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack
spec:
  jobLabel: ray-workers
  namespaceSelector:
    matchNames: [default]
  selector:
    matchLabels:
      ray.io/node-type: worker
  podMetricsEndpoints:
    - port: metrics
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_label_ray_io_cluster]
          targetLabel: ray_io_cluster
```

```bash
kubectl apply -f 800-ray/ray-podmonitor.yaml
kubectl get podmonitor -n monitoring | grep ray
```

```
podmonitor.monitoring.coreos.com/ray-head-monitor created
podmonitor.monitoring.coreos.com/ray-workers-monitor created
ray-head-monitor      7s
ray-workers-monitor   7s
```

Three details carry over from [Chapter 03](03-observability.md):

- **`PodMonitor`, not `ServiceMonitor`.** Ray metrics are per-pod, and you need head and worker
  separated. A `ServiceMonitor` behind a load-balanced Service would mix them.
- **`release: kube-prometheus-stack` again.** Same silent failure mode: without it the
  `PodMonitor` exists and is never scraped.
- **The `relabelings` add `ray_io_cluster`.** Every Ray dashboard panel filters on it. Without
  the relabel, metrics from multiple Ray clusters in one namespace are indistinguishable and
  the dashboards render empty.

### The dashboards

```bash
kubectl get grafanadashboards -n monitoring | grep -E 'ray|inference'
```

```
ray-grafana-default-dashboard              <resync>   <age>
ray-grafana-inference-overview-dashboard   <resync>   <age>
ray-grafana-serve-dashboard                <resync>   <age>
ray-grafana-serve-deployment-dashboard     <resync>   <age>
vllm-ray-grafana-dashboard                 <resync>   <age>
```

Five `GrafanaDashboard` CRs, each answering a different layer's question:

| Dashboard | Layer |
|---|---|
| `ray-grafana-default-dashboard` | Ray cluster — nodes, CPU/GPU/memory, object store |
| `ray-grafana-serve-dashboard` | Ray Serve — request rate, latency, errors per application |
| `ray-grafana-serve-deployment-dashboard` | Per-deployment replicas, queue depth |
| `vllm-ray-grafana-dashboard` | vLLM engine metrics under Ray (the Chapter 03 panels) |
| `ray-grafana-inference-overview-dashboard` | Rolled-up inference view |

These appeared in the `Grafana external-grafana` CR's `status.dashboards` list back in
[Chapter 03](03-observability.md#whats-already-installed) — declared as Kubernetes objects and
reconciled into Grafana automatically. Reuse the same Grafana Ingress:

```bash
kubectl apply -f 100-vllm/grafana-ingress.yaml
export GRAFANA_URL=$(kubectl get ingress/grafana-ingress -n monitoring \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Your Grafana URL is: http://$GRAFANA_URL"

export GRAFANA_PASSWORD=$(kubectl get secret -n monitoring kube-prometheus-stack-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode)
echo "  Username: admin"
echo "  Password: ${GRAFANA_PASSWORD}"
```

```
ingress.networking.k8s.io/grafana-ingress unchanged
Your Grafana URL is: http://<ALB_DNS>

Grafana Credentials:
  Username: admin
  Password: <REDACTED>
```

The observability layer is **completely unchanged** across serving frameworks. Prometheus
Operator CRs plus grafana-operator CRs meant switching from a Deployment to Ray Serve cost two
`PodMonitor`s. That's the payoff for treating monitoring as Kubernetes objects rather than
per-app configuration.

## So: Ray Serve or a plain Deployment?

| | Deployment + vLLM (Ch 02) | RayService + KubeRay (Ch 09) |
|---|---|---|
| Objects to understand | Deployment, Service, Ingress | RayService → RayCluster → head/worker pods → Serve app |
| Service ports | 1 | 8 |
| Time to ready | seconds after node + image | **> 10 min, twice** |
| Image | upstream vLLM | AWS DLC (version matrix pre-solved) |
| Extra operator | none | KubeRay |
| Model parallel across nodes | no | **yes** |
| Multi-model / composed pipelines | you build it | **Ray Serve primitives** |
| Built-in per-replica UI | no | **Ray dashboard** |
| Failure surface | pod | pod, GCS, actors, Serve controller |

**Use a plain Deployment when** one model fits on one GPU — which covered every other chapter
in this tutorial. It is simpler in every dimension and there is no performance penalty.

**Reach for Ray when you actually need what Ray does:** a model too large for one node
(tensor/pipeline parallel across machines), a pipeline of several models with different scaling
behaviour, or Ray Serve's composition and autoscaling primitives. If you're using Ray only to
run one vLLM on one GPU, you've bought a distributed scheduler and paid for it in ports,
startup time, and failure modes.

That's not a criticism of Ray — it's the standard advice about distributed systems, and the
run in this chapter is a concrete illustration of the cost side.

Next: [10 — Cleanup](10-cleanup.md)
