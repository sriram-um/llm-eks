# 10 — Cleanup

An L40S GPU node, three ALBs, an ElastiCache Serverless cache and an On-Demand Capacity
Reservation all bill by the hour whether or not anything is using them. The ODCR is the worst
offender: **it charges even with zero instances running.**

Do this in order — workloads, then operators, then infrastructure — so that finalizers and
controllers can clean up their AWS resources before you remove the controllers themselves.

> **In plain terms — why the order matters.** Several AWS resources in this tutorial were never
> created by you directly. You created an `Ingress`, and the AWS Load Balancer **Controller**
> went and built a real ALB for you. You created a `NodePool`, and **Karpenter** launched an EC2
> instance.
>
> Those controllers are also the only things that know how to *delete* what they made. Kubernetes
> uses **finalizers** — a "don't actually remove this object until I've finished cleaning up"
> marker — to make that work.
>
> So if you delete the controller first, nobody is left to hear the delete:
>
> ```
>   ✗ remove LB controller → then delete Ingress
>       → the ALB keeps existing, keeps billing, and is now invisible to kubectl
>
>   ✓ delete Ingress → controller deletes the ALB → then remove the controller
> ```
>
> An orphaned load balancer or GPU node is a real and fairly common way to be surprised by a
> bill, and it's specifically what "workloads, then operators, then infrastructure" prevents.
>
> The **capacity reservation** deserves its own note because it breaks the usual intuition: it is
> a *promise* that a GPU will be available for you, and you pay for that promise whether or not
> an instance exists. Deleting every pod and node in the cluster does **not** stop it. Cancelling
> it is a separate, explicit step at the end of this chapter — the single most important command
> on this page.
>
> Finally, the last section verifies rather than assumes. Deletes can silently do nothing (a
> resource in another region, a name that no longer matches), so it lists what's actually left.

## 1. Delete the workloads

```bash
# 100-serve-model
kubectl delete deployment mistral 2>/dev/null || true
kubectl delete deployment strands-weather-agent 2>/dev/null || true
kubectl delete service strands-weather-agent 2>/dev/null || true

# KV cache labs
kubectl delete pod lmcache-cpu-ram 2>/dev/null || true
kubectl delete configmap cpu-ram-cm 2>/dev/null || true
kubectl delete pod lmcache-valkey 2>/dev/null || true
kubectl delete configmap valkey-cm 2>/dev/null || true

# Benchmark Helm releases + engine metrics dashboard
helm uninstall baseline -n default 2>/dev/null || true
helm uninstall saturation -n default 2>/dev/null || true
kubectl delete grafanadashboard vllm-benchmarking-dashboard -n monitoring 2>/dev/null || true
kubectl delete configmap vllm-benchmarking-dashboard-config -n monitoring 2>/dev/null || true

# Ray Serve + Open WebUI
kubectl delete -f 800-ray/ray-vllm-service.yaml 2>/dev/null || true
kubectl delete rayservice vllm -n default 2>/dev/null || true
kubectl delete -f 800-ray/openwebui.yml 2>/dev/null || true
kubectl delete deployment open-webui 2>/dev/null || true
kubectl delete service open-webui 2>/dev/null || true
kubectl delete ingress open-webui-ingress 2>/dev/null || true
kubectl delete podmonitor ray-head-monitor -n monitoring 2>/dev/null || true
kubectl delete podmonitor ray-workers-monitor -n monitoring 2>/dev/null || true

# Fine-tuning
kubectl delete deployment anyvc 2>/dev/null || true
kubectl delete configmap anyvc-training-scripts 2>/dev/null || true
kubectl delete job anyvc-lora-finetune 2>/dev/null || true

# RAG
kubectl delete deployment rag-service 2>/dev/null || true
kubectl delete service rag-service 2>/dev/null || true
kubectl delete configmap rag-serve-script 2>/dev/null || true
kubectl delete configmap rag-processor-script 2>/dev/null || true
kubectl delete deployment rag-gradio-interface 2>/dev/null || true
kubectl delete service rag-gradio-interface 2>/dev/null || true
kubectl delete ingress rag-gradio-alb 2>/dev/null || true
kubectl delete configmap rag-gradio-app 2>/dev/null || true
kubectl delete job rag-document-processor 2>/dev/null || true

# Shared Service, PVC, PV
kubectl delete service vllm-serve-svc 2>/dev/null || true
kubectl delete pvc s3-bucket-pvc 2>/dev/null || true
kubectl delete pv s3-bucket-pv 2>/dev/null || true

# GPU capacity test
kubectl delete pod nvidia-smi 2>/dev/null || true

# Observability
kubectl delete servicemonitor mistral-monitor -n monitoring 2>/dev/null || true
kubectl delete -f 100-vllm/grafana-ingress.yaml 2>/dev/null || true
helm uninstall dcgm-exporter -n monitoring 2>/dev/null || true
```

```
grafanadashboard.grafana.integreatly.org "vllm-benchmarking-dashboard" deleted from monitoring namespace
configmap "vllm-benchmarking-dashboard-config" deleted from monitoring namespace
rayservice.ray.io "vllm" deleted from default namespace
deployment.apps "open-webui" deleted from default namespace
service "open-webui" deleted from default namespace
ingress.networking.k8s.io "open-webui-ingress" deleted from default namespace
podmonitor.monitoring.coreos.com "ray-head-monitor" deleted from monitoring namespace
podmonitor.monitoring.coreos.com "ray-workers-monitor" deleted from monitoring namespace
configmap "anyvc-training-scripts" deleted from default namespace
persistentvolumeclaim "s3-bucket-pvc" deleted from default namespace
persistentvolume "s3-bucket-pv" deleted
pod "nvidia-smi" deleted from default namespace
servicemonitor.monitoring.coreos.com "mistral-monitor" deleted from monitoring namespace
ingress.networking.k8s.io "grafana-ingress" deleted from monitoring namespace
release "dcgm-exporter" uninstalled
```

The `2>/dev/null || true` on every line is deliberate: this script is meant to be safe to run
at any point, including after you've already deleted half of it. Most objects in the output
above were already gone from earlier chapters' teardowns, and the script neither errors nor
stops.

**Delete Ingresses, not just Deployments.** Each ALB Ingress in this tutorial
(`open-webui-ingress`, `grafana-ingress`, `rag-gradio-alb`) has a real Application Load
Balancer behind it. Deleting the Deployment leaves the ALB running and billing. The AWS Load
Balancer Controller only removes the ALB when the `Ingress` object goes away — which is also
why the controller must still be running when you delete them.

Verify:

```bash
kubectl get ingress -A
aws elbv2 describe-load-balancers --query 'LoadBalancers[].DNSName' --output table
```

Both should be empty. If an ALB lingers, the Ingress finalizer is probably still pending —
check `kubectl get ingress -A -o yaml | grep finalizers -A 3`.

## 2. Uninstall the KubeRay operator

Operators go **after** their custom resources. Removing KubeRay while a `RayService` still
exists leaves orphaned pods that nothing will reap.

```bash
helm uninstall kuberay-operator 2>/dev/null || true

kubectl wait --for=delete pods -l app.kubernetes.io/name=kuberay-operator --timeout=300s \
  2>/dev/null || true
```

```
release "kuberay-operator" uninstalled
```

## 3. Confirm the GPU node goes away

Karpenter deprovisions nodes once nothing schedules on them. This is the check that actually
stops the expensive meter:

```bash
kubectl get nodes -l karpenter.sh/nodepool=gpu
kubectl get nodeclaims
```

Expect `No resources found` within a few minutes. If a GPU node persists, something is still
scheduled on it:

```bash
kubectl get pods -A -o wide | grep <node-name>
```

## 4. Cancel the On-Demand Capacity Reservation

This step matters most and has the ugliest failure mode.

### What the workshop instructions do — and why it breaks

```bash
# DON'T DO THIS
AWS_REGION=$(kubectl config get-contexts | grep '*' | awk '{print $2}' | cut -d':' -f4)
echo "Using AWS Region: $AWS_REGION"

ODCR_ID=$(aws ec2 describe-capacity-reservations --region $AWS_REGION \
  --filters "Name=state,Values=active" \
  --query "CapacityReservations[0].CapacityReservationId" --output text)
echo "Found ODCR ID: $ODCR_ID"

aws ec2 cancel-capacity-reservation --region $AWS_REGION --capacity-reservation-id $ODCR_ID
```

```
Using AWS Region: genai-workshop

aws: [ERROR]: Could not connect to the endpoint URL: "https://ec2.genai-workshop.amazonaws.com/"
Found ODCR ID: 

aws: [ERROR]: An error occurred (ParamValidation): argument --capacity-reservation-id: expected one argument
```

The region extraction assumes the kubectl context name is a cluster ARN
(`arn:aws:eks:us-east-2:123456789012:cluster/name`), so field 4 of the colon-split is the
region. This context was named `genai-workshop`, so `cut -d':' -f4` returned the context name
itself. `describe-capacity-reservations` then failed against a nonexistent endpoint, `ODCR_ID`
came back empty, and `cancel-capacity-reservation` failed on the missing argument.

**Nothing was cancelled, and both errors are easy to skim past.** See
[GOTCHAS](GOTCHAS.md#odcr-cleanup-derives-the-wrong-region).

### Do this instead

```bash
export AWS_REGION=$(aws configure get region)
# or: export AWS_REGION=us-east-2
echo "Using AWS Region: $AWS_REGION"

aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
  --filters "Name=state,Values=active" \
  --query 'CapacityReservations[].{Id:CapacityReservationId,Type:InstanceType,
            Total:TotalInstanceCount,Available:AvailableInstanceCount,AZ:AvailabilityZone}' \
  --output table
```

Read the table before you cancel anything — `CapacityReservations[0]` in the original snippet
blindly takes the first active reservation in the account, which may not be yours. Then cancel
by explicit ID:

```bash
aws ec2 cancel-capacity-reservation \
  --region "$AWS_REGION" \
  --capacity-reservation-id cr-0123456789abcdef0
```

Confirm it's gone:

```bash
aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
  --filters "Name=state,Values=active" --output table
```

## 5. Delete the ElastiCache Serverless cache

Created for [Chapter 06](06-kv-cache-offloading.md)'s L2 cache and not removed by anything above.

```bash
aws elasticache delete-serverless-cache \
  --serverless-cache-name lmcache-valkey-eks \
  --region "$AWS_REGION"

aws elasticache describe-serverless-caches --region "$AWS_REGION" --output table
```

## 6. Empty the S3 buckets (optional)

Storage is cheap relative to a GPU, but the model weights are 10.4 GB and the benchmark results
accumulate on every run.

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Model weights, LoRA adapter, benchmark results
aws s3 rm "s3://genai-models-${AWS_ACCOUNT_ID}/benchmarks" --recursive
aws s3 rm "s3://genai-models-${AWS_ACCOUNT_ID}/anyvc-startup-lora" --recursive
# aws s3 rm "s3://genai-models-${AWS_ACCOUNT_ID}/Ministral-3-8B-Instruct-2512" --recursive

# RAG source documents
aws s3 rm "s3://rag-workshop-data-${AWS_ACCOUNT_ID}-${AWS_REGION}/samples" --recursive
```

Keep the model prefix if you plan to run any of this again — re-downloading it took **13
minutes** in [Chapter 00](00-environment-and-cluster.md).

The S3 **vector** bucket needs the `s3vectors` API, not `s3`:

```bash
aws s3vectors delete-index \
  --vector-bucket-name "rag-vectors-${AWS_ACCOUNT_ID}-${AWS_REGION}" \
  --index-name knowledge-base

aws s3vectors delete-vector-bucket \
  --vector-bucket-name "rag-vectors-${AWS_ACCOUNT_ID}-${AWS_REGION}"
```

## 7. Destroy the cluster

If you provisioned the cluster yourself with Terraform:

```bash
cd sample-genai-on-eks/terraform
terraform destroy --auto-approve
```

In a **Workshop Studio** environment this will not work, and shouldn't:

```
bash: cd: sample-genai-on-eks/terraform: No such file or directory
bash: terraform: command not found
```

The workshop account tears down its own infrastructure when the event ends — the Terraform
state lives outside your IDE environment. If that's your situation, steps 1–6 are all you
control, and step 4 (the ODCR) is the one that costs real money if you skip it.

## Final verification

```bash
kubectl get nodes -l karpenter.sh/nodepool=gpu           # → No resources found
kubectl get ingress -A                                   # → No resources found
aws elbv2 describe-load-balancers --query 'length(LoadBalancers)'
aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
  --filters "Name=state,Values=active" \
  --query 'length(CapacityReservations)'
aws elasticache describe-serverless-caches --region "$AWS_REGION" \
  --query 'length(ServerlessCaches)'
```

Four zeros and no GPU nodes means you're done.

---

Back to the [index](README.md) · [GOTCHAS](GOTCHAS.md)
