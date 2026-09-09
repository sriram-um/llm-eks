# 01 — GPU capacity

Goal: understand the GPU `NodePool`/`NodeClass` pair, then prove a GPU node can actually be
provisioned *before* you wait 10 minutes for a model server to fail to schedule.

> **In plain terms.** There is no GPU in the cluster yet, and that's deliberate. A **NodePool**
> is a *policy* — "you're allowed to buy this kind of machine, with these labels and these
> restrictions" — and a **NodeClass** is the *recipe* for building one (which disk, which image,
> which capacity reservation). Nothing is running until a pod asks for a GPU.
>
> Getting a pod onto that GPU then requires **three independent things to line up**, and they
> sound redundant but aren't:
>
> 1. `nodeSelector` — *you* pick the node type ("I want a GPU node").
> 2. `tolerations` — the *node* admits you. GPU nodes carry a `NoSchedule` **taint** that repels
>    ordinary pods, so a random web frontend can't squat on a $2/hour card.
> 3. `resources.limits: nvidia.com/gpu: 1` — the device plugin hands you an actual GPU.
>
> Miss #1 and you quietly land on a CPU node. Miss #2 and you sit in `Pending` forever with no
> error. Which is exactly why this chapter's payload is a **throwaway pod that just runs
> `nvidia-smi`** — a 30-second experiment that proves all three work before you ask a 10 GB
> model server to schedule and spend ten minutes wondering why nothing happened.
>
> One more thing worth knowing up front: modern GPUs are genuinely scarce, so this chapter uses
> an **On-Demand Capacity Reservation** to guarantee a card is available. That reservation bills
> whether or not you use it — see [Chapter 10](10-cleanup.md).
>
> *Background:* [CONCEPTS Part 3](CONCEPTS.md#part-3--why-a-gpu-and-why-utilization-is-the-entire-cost-story)
> (why a GPU at all) and
> [Part 5](CONCEPTS.md#part-5--karpenter-and-eks-auto-mode-nodes-that-appear-when-needed)
> (how nodes appear on demand).

## Inspect the GPU NodePool and NodeClass

```bash
kubectl get nodepool/gpu
kubectl get nodeclass/gpu
```

```
NAME   NODECLASS   NODES   READY   AGE
gpu    gpu         0       True    <age>

NAME   ROLE                          READY   AGE
gpu    <cluster>-eks-node-role-…     True    <age>
```

`NODES 0` — nothing is running yet. Karpenter provisions on demand, so the GPU node (and its
bill) only exists while a pod needs it.

The interesting part is in the `NodeClass`:

```bash
kubectl get nodeclass gpu -o yaml | grep -A 3 capacityReservationSelectorTerms
```

```yaml
  capacityReservationSelectorTerms:
  - id: <ODCR_ID>          # e.g. cr-0123456789abcdef0
  ephemeralStorage:
    iops: 3000
```

Two decisions worth copying into your own clusters:

**1. `capacityReservationSelectorTerms`.** Karpenter is pinned to a specific EC2 On-Demand
Capacity Reservation. L40S / H100 / A100 capacity is genuinely scarce in most regions —
without an ODCR your NodePool can sit at `NODES 0` indefinitely while pods stay `Pending`
with an insufficient-capacity event. Reserving first and pointing Karpenter at the
reservation converts "maybe I get a GPU" into "I get a GPU". It also means you are paying
for the reservation whether or not a node is running, which is exactly why Chapter 10 ends
with `aws ec2 cancel-capacity-reservation`.

**2. Tuned `ephemeralStorage`.** 3000 IOPS on the root volume. Container images for GPU
inference are enormous (the vLLM image plus CUDA runtime is multiple GB) and SOCI lazy
loading is I/O-bound, so provisioned IOPS directly shortens cold-start.

## Smoke-test the GPU with a throwaway pod

Do not debug GPU scheduling through a model server. Use the smallest possible pod:

```bash
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  nodeSelector:
    karpenter.sh/nodepool: gpu
  containers:
  - name: nvidia-smi
    image: public.ecr.aws/amazonlinux/amazonlinux:2023-minimal
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: OnFailure
EOF
```

```
pod/nvidia-smi created
```

```bash
# Blocks until the pod reaches Succeeded
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/nvidia-smi --timeout=300s
```

```
pod/nvidia-smi condition met
```

Three things had to all be true for that one line to print, and each is a common failure
mode:

| Requirement | What it proves |
|---|---|
| `nodeSelector: karpenter.sh/nodepool: gpu` | Karpenter selected the GPU NodePool and got capacity from the ODCR |
| `tolerations` for `nvidia.com/gpu:NoSchedule` | GPU nodes are tainted so CPU workloads can't squat on them; your pod must opt in |
| `limits: nvidia.com/gpu: 1` | The device plugin is present and advertising the GPU as a schedulable resource |

Note that the image is `amazonlinux:2023-minimal` — a plain base image with no CUDA
libraries. `nvidia-smi` works because EKS Auto Mode's GPU AMI injects the driver and the
NVIDIA container runtime hooks. If you can run `nvidia-smi` from an arbitrary image, the
node-level GPU plumbing is correct and any later failure is your application's.

## What you got

Later chapters (via DCGM Exporter, Chapter 03) reveal the hardware:

```
modelName="NVIDIA L40S"
UUID="GPU-…"
hostname="i-<gpu-node-id>"
DCGM_FI_DEV_FB_FREE   4895   # MiB
DCGM_FI_DEV_FB_USED  40562   # MiB
DCGM_FI_DEV_SM_CLOCK  2520   # MHz
```

One L40S, ~45.5 GiB of usable framebuffer. Budget for it: an 8.9 B-parameter model in BF16
occupies ~17.8 GB of weights, and vLLM claims most of the rest as KV cache
(27.91 GiB — see Chapter 02).

Leave the pod in place or delete it; it costs nothing once `Succeeded`, but the **node** it
provisioned stays warm for a while, which conveniently means the next chapter's vLLM pod
starts on an already-running GPU node instead of waiting for a fresh one.

Next: [02 — Serve the model with vLLM](02-serve-with-vllm.md)
