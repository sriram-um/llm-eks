# 00 — Environment and cluster

Goal: confirm the cluster is healthy, understand how EKS Auto Mode partitions workloads
across NodePools, and lock down the ALB before anything is exposed.

> **In plain terms.** A Kubernetes cluster is a pool of machines plus a scheduler that decides
> what runs where. You never say "start this container on that server" — you submit a document
> describing what you want to exist, and controllers keep reality matching it.
>
> **EKS Auto Mode** goes one step further: AWS also decides *which machines exist at all*. The
> three **NodePools** you'll see below are three tiers of hardware with different prices and
> capabilities — cheap CPU nodes for the control-plane add-ons, general CPU nodes for web apps,
> and (added in Chapter 01) expensive GPU nodes for the model. Nodes get created when a pod
> needs one and deleted when nothing does, which is why a GPU NodePool with **zero nodes** is
> the correct, non-broken, non-billing idle state.
>
> The last section of this chapter is the one to not skip: it closes the front door (locks the
> load balancer to your own IP) *before* anything is exposed to the internet.
>
> *New to Kubernetes?* Read
> [CONCEPTS Part 4](CONCEPTS.md#part-4--kubernetes-only-the-parts-this-tutorial-uses) and
> [Part 5](CONCEPTS.md#part-5--karpenter-and-eks-auto-mode-nodes-that-appear-when-needed)
> first — about six minutes, and everything after here will read normally.

## Verify the cluster

```bash
kubectl get pods --all-namespaces
```

```
NAMESPACE     NAME                                               READY   STATUS       RESTARTS   AGE
default       model-download-…                                   0/1     Init:Error   0          <age>
default       model-download-…                                   0/1     Completed    0          <age>
kube-system   metrics-server-…                                   1/1     Running      0          <age>
kube-system   s3-csi-controller-…                                1/1     Running      0          <age>
kube-system   s3-csi-node-…                                      3/3     Running      0          <age>
kube-system   s3-csi-node-…                                      3/3     Running      0          <age>
monitoring    grafana-operator-…                                 1/1     Running      0          <age>
monitoring    kube-prometheus-stack-grafana-…                    3/3     Running      0          <age>
monitoring    kube-prometheus-stack-kube-state-metrics-…         1/1     Running      0          <age>
monitoring    kube-prometheus-stack-operator-…                   1/1     Running      0          <age>
monitoring    kube-prometheus-stack-prometheus-node-exporter-…   1/1     Running      0          <age>
monitoring    prometheus-kube-prometheus-stack-prometheus-0      2/2     Running      0          <age>
```

Three things to read out of this:

1. **`s3-csi-controller` / `s3-csi-node`** — the Mountpoint for Amazon S3 CSI driver is
   installed. Chapter 07 uses it to mount the model bucket into a vLLM pod as a
   `PersistentVolume` so a LoRA adapter can be read from disk.
2. **`monitoring` namespace** — kube-prometheus-stack *and* the grafana-operator are both
   pre-installed. The operator matters: dashboards are created as `GrafanaDashboard` CRs,
   not by clicking around in the UI.
3. **One `model-download` pod is `Init:Error`, the other is `Completed`.** The
   Job succeeded on retry. This is normal and not a problem — see
   [GOTCHAS](GOTCHAS.md#model-download-job-shows-initerror) for why you should check the
   Job, not the pods.

```bash
kubectl get job model-download -o wide
```

```
NAME             STATUS     COMPLETIONS   DURATION   AGE     CONTAINERS   IMAGES
model-download   Complete   1/1           13m        <age>   download     python:3.11-slim
```

13 minutes to pull ~10.5 GB of weights from Hugging Face into S3, once, on a general-purpose
node. Every later chapter reads from S3 instead of the internet.

## Lock the ALB to your IP before exposing anything

EKS Auto Mode's ALB integration is configured through an `IngressClassParams` object. Patch
it once and *every* Ingress created with `ingressClassName: alb` inherits the source-IP
restriction — even though the individual Ingress manifests in this repo say
`inbound-cidrs: 0.0.0.0/0`.

```bash
export MY_IP=<YOUR_IP>
kubectl patch ingressclassparams alb --type=merge \
  -p "{\"spec\":{\"inboundCIDRs\":[\"${MY_IP}/32\"]}}"
```

```
ingressclassparams.eks.amazonaws.com/alb patched
```

Do this **first**. Three internet-facing ALBs get created over the course of this tutorial
(Open WebUI, Grafana, the Gradio RAG UI), and one of them fronts an unauthenticated LLM.

## How Auto Mode partitions the cluster

Auto Mode ships with two built-in Karpenter NodePools, and this workshop adds a third.

```bash
kubectl get nodes -l karpenter.sh/nodepool=general-purpose
kubectl get nodes -l karpenter.sh/nodepool=system
```

```
NAME                  STATUS   ROLES    AGE     VERSION
i-<general-node-id>   Ready    <none>   <age>   v1.36.2-eks-0690643

NAME                  STATUS   ROLES    AGE     VERSION
i-<system-node-id>    Ready    <none>   <age>   v1.36.2-eks-0690643
```

| NodePool | Purpose | Who lands there |
|---|---|---|
| `system` | cluster add-ons | Prometheus, Grafana, grafana-operator, kube-state-metrics |
| `general-purpose` | CPU workloads | model-download Job, Open WebUI, Gradio, RAG service, node-exporter |
| `gpu` (custom) | accelerated workloads | vLLM, DCGM Exporter, LoRA training Job, Ray workers |

Confirm the placement rather than assuming it:

```bash
kubectl get pods -n monitoring \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}' \
| while read pod node; do
    echo "Pod: $pod -> Node: $node ($(kubectl get node $node \
      -o jsonpath='{.metadata.labels.karpenter\.sh/nodepool}' 2>/dev/null || echo unknown))"
  done
```

```
Pod: grafana-operator-…                              -> Node: i-<system-node-id>  (system)
Pod: kube-prometheus-stack-grafana-…                 -> Node: i-<system-node-id>  (system)
Pod: kube-prometheus-stack-kube-state-metrics-…      -> Node: i-<system-node-id>  (system)
Pod: kube-prometheus-stack-operator-…                -> Node: i-<system-node-id>  (system)
Pod: kube-prometheus-stack-prometheus-node-exporter-… -> Node: i-<general-node-id> (general-purpose)
Pod: prometheus-kube-prometheus-stack-prometheus-0   -> Node: i-<system-node-id>  (system)
```

The whole observability stack sits on the `system` pool. That is deliberate: you never want
Prometheus competing with a model server for a $2/hour GPU node's CPU, and you want metrics
to survive GPU node churn.

## Shared environment variables

Nearly every manifest in the repo is templated with `envsubst`, so export these once per
shell:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="genai-models-${AWS_ACCOUNT_ID}"

echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "S3 Bucket: $S3_BUCKET_NAME"
```

```
AWS Account ID: <ACCOUNT_ID>
AWS Region: us-east-2
S3 Bucket: genai-models-<ACCOUNT_ID>
```

Verify the model is where you think it is:

```bash
aws s3 ls s3://${S3_BUCKET_NAME}/Ministral-3-8B-Instruct-2512/ --recursive
```

```
      20311 Ministral-3-8B-Instruct-2512/README.md
       2361 Ministral-3-8B-Instruct-2512/SYSTEM_PROMPT.txt
       1903 Ministral-3-8B-Instruct-2512/config.json
10420633176 Ministral-3-8B-Instruct-2512/consolidated.safetensors
        131 Ministral-3-8B-Instruct-2512/generation_config.json
     103195 Ministral-3-8B-Instruct-2512/model.safetensors.index.json
       1185 Ministral-3-8B-Instruct-2512/params.json
        976 Ministral-3-8B-Instruct-2512/processor_config.json
   16753784 Ministral-3-8B-Instruct-2512/tekken.json
   17077420 Ministral-3-8B-Instruct-2512/tokenizer.json
      21177 Ministral-3-8B-Instruct-2512/tokenizer_config.json
```

**10,420,633,176 bytes** in a single `consolidated.safetensors`. Remember that number — the
next two chapters are largely about not paying for it twice.

Next: [01 — GPU capacity](01-gpu-capacity.md)
