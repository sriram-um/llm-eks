# 08 — RAG with Amazon S3 Vectors

Chapter 07 changed *how* the model answers. This chapter changes *what it knows*, with no
training at all: retrieve relevant documents at query time and put them in the prompt.

The vector store is **Amazon S3 Vectors** — vector search as an S3 bucket type. No OpenSearch
domain, no pgvector instance, no Pinecone account, nothing to right-size or patch.

> **In plain terms.** RAG turns a closed-book exam into an open-book one. Before answering, you
> look up the relevant pages and paste them into the prompt. The model isn't smarter — it just
> has the material in front of it.
>
> The only genuinely new idea is **how you find the right pages**, because keyword search fails
> here: a user asking *"how long does it last on one charge?"* needs a document that says
> *"battery life: 12 hours"*, and those share no words at all.
>
> The fix is **embeddings**. An embedding model converts text into a list of numbers — 384 of
> them here — positioned so that similar *meaning* ends up in a similar place. Meaning becomes
> geometry, and "find related text" becomes "find nearby points":
>
> ```
>   "how long on one charge"  → [0.02, -0.41, 0.88, …] ┐ nearly the same direction
>   "battery life: 12 hours"  → [0.04, -0.38, 0.91, …] ┘ → high similarity
>   "speaker sound quality"   → [0.77,  0.12, -0.03, …]  → low similarity
> ```
>
> Similarity is the **angle** between the vectors: 1.0 identical, 0 unrelated. You'll also see
> **cosine distance**, which is `1 − similarity`, so **lower is closer**. That sign flip catches
> everyone exactly once.
>
> The whole pipeline is two loops:
>
> ```
>   INGEST (once)   documents → split into chunks → embed → store vectors
>   QUERY  (each)   question → embed → nearest vectors → paste text into prompt → ask the LLM
> ```
>
> Two results below are worth watching for. First, a **deliberately fake query vector of all
> `0.1`s returns distances of ~0.978 against everything** — the index answers happily, and the
> ranking is meaningless. A query returning results is *not* evidence that retrieval works; you
> have to look at the scores. Second, retrieval added **220 ms to a 15-second request — 1.4 %.**
> Retrieval is essentially free and generation is the entire cost, so the only interesting
> question in a RAG system is whether your chunks are any good.
>
> *Background:* [CONCEPTS Part 9](CONCEPTS.md#part-9--embeddings-and-vector-search-intuitively),
> and [Part 8](CONCEPTS.md#part-8--three-ways-to-change-what-a-model-does) for RAG vs fine-tuning.

## Configure

```bash
echo "Using AWS Region: $AWS_REGION"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

export S3_BUCKET_NAME="rag-workshop-data-${ACCOUNT_ID}-${AWS_REGION}"
export S3_VECTOR_BUCKET_NAME="rag-vectors-${ACCOUNT_ID}-${AWS_REGION}"
export S3_VECTOR_INDEX_NAME="knowledge-base"
```

```
AWS Region: <AWS_REGION>
AWS Account ID: <ACCOUNT_ID>
S3 Data Bucket: rag-workshop-data-<ACCOUNT_ID>-<AWS_REGION>
S3 Vector Bucket: rag-vectors-<ACCOUNT_ID>-<AWS_REGION>
S3 Vector Index: knowledge-base
```

Two separate buckets, deliberately: a **general-purpose bucket** for source documents and a
**vector bucket** for embeddings. Different bucket types, different APIs (`aws s3` vs
`aws s3vectors`).

```bash
aws s3vectors list-vector-buckets
aws s3vectors list-indexes --vector-bucket-name $S3_VECTOR_BUCKET_NAME
kubectl get serviceaccount s3-access-sa -n default
```

```json
{
    "vectorBuckets": [
        {
            "vectorBucketName": "rag-vectors-<ACCOUNT_ID>-<AWS_REGION>",
            "vectorBucketArn": "arn:aws:s3vectors:<AWS_REGION>:<ACCOUNT_ID>:bucket/rag-vectors-<ACCOUNT_ID>-<AWS_REGION>",
            "creationTime": "<timestamp>"
        }
    ]
}
{
    "indexes": [
        {
            "vectorBucketName": "rag-vectors-<ACCOUNT_ID>-<AWS_REGION>",
            "indexName": "knowledge-base",
            "indexArn": "arn:aws:s3vectors:<AWS_REGION>:<ACCOUNT_ID>:bucket/rag-vectors-<ACCOUNT_ID>-<AWS_REGION>/index/knowledge-base",
            "creationTime": "<timestamp>"
        }
    ]
}
NAME           AGE
s3-access-sa   <age>
```

`s3-access-sa` is an EKS Pod Identity service account. The in-cluster pods call
`s3vectors:PutVectors` / `QueryVectors` with no credentials in any manifest — same pattern as
`model-storage-sa` in [Chapter 05](05-benchmarking.md).

## Ingest: documents → embeddings → vectors

```bash
aws s3 cp 700-rag/electronics.jsonl s3://${S3_BUCKET_NAME}/samples/electronics.jsonl
aws s3 ls s3://${S3_BUCKET_NAME}/samples/
```

```
upload: 700-rag/electronics.jsonl to s3://rag-workshop-data-<ACCOUNT_ID>-<AWS_REGION>/samples/electronics.jsonl
<date> <time>      22458 electronics.jsonl
```

22 KB — 25 product reviews. Small on purpose: the mechanics are identical at 25 records and
25 million, and you can read the whole corpus to verify what retrieval returns.

```bash
kubectl apply -f 700-rag/rag-processor.yml

envsubst < 700-rag/rag-document-job.yml > 700-rag/rag-document-job-processed.yml
grep -A 5 S3_VECTOR_BUCKET_NAME 700-rag/rag-document-job-processed.yml
kubectl apply -f 700-rag/rag-document-job-processed.yml
```

```
configmap/rag-processor-script created
        - name: S3_VECTOR_BUCKET_NAME
          value: "rag-vectors-<ACCOUNT_ID>-<AWS_REGION>"
        - name: S3_VECTOR_INDEX_NAME
          value: "knowledge-base"
        - name: S3_BUCKET_NAME
          value: "rag-workshop-data-<ACCOUNT_ID>-<AWS_REGION>"
job.batch/rag-document-processor created
```

```bash
kubectl get jobs rag-document-processor
export POD_NAME=$(kubectl get pods --selector=job-name=rag-document-processor \
  -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $POD_NAME
```

```
NAME                     STATUS    COMPLETIONS   DURATION   AGE
rag-document-processor   Running   0/1           38s        38s

Successfully installed boto3-1.43.43 botocore-1.43.90 …
Looking in indexes: https://download.pytorch.org/whl/cpu
Successfully installed … torch-2.13.0+cpu …
Successfully installed … sentence-transformers-5.6.0 transformers-5.16.1 …

Using S3 bucket: rag-workshop-data-<ACCOUNT_ID>-<AWS_REGION>
Using S3 Vector Bucket: rag-vectors-<ACCOUNT_ID>-<AWS_REGION>
Using S3 Vector Index: knowledge-base
Loading SentenceTransformer model...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 4688.76it/s]
Model loaded successfully
Connecting to S3 Vectors...
Connected to S3 Vectors
Starting S3 Vectors document processing pipeline...
Finding documents in S3 bucket: rag-workshop-data-<ACCOUNT_ID>-<AWS_REGION>...
Found document: samples/electronics.jsonl
Found 1 documents in S3
Starting document processing...
Downloaded file size: 22458 bytes
Processed 25 records from file: /tmp/electronics.jsonl
Final batch: Uploaded 25 vectors to S3 Vectors
Total embeddings generated and stored: 25
Verification query returned 1 results
 S3 Vectors storage verification successful!
S3 Vectors document processing complete!
```

`Looking in indexes: https://download.pytorch.org/whl/cpu` — the embedding job installs
**CPU-only PyTorch** and runs `all-MiniLM-L6-v2` (a 22 M-parameter model) on a CPU node. The
GPU is not involved in ingestion. Sizing the hardware to the model, not to the pipeline, is the
whole point of splitting these into separate workloads.

The job also verifies its own work (`Verification query returned 1 results`) before exiting.
An ingest job that silently writes zero vectors is a classic RAG failure — it looks fine until
every answer is unsourced.

## Verify the index directly

```bash
python3 -c "
import json
test_vector = [0.1] * 384          # 384 dims = all-MiniLM-L6-v2
query_vector = {'float32': test_vector}
print(json.dumps(query_vector))
" > 700-rag/test_query_384.json

aws s3vectors query-vectors \
    --vector-bucket-name $S3_VECTOR_BUCKET_NAME \
    --index-name $S3_VECTOR_INDEX_NAME \
    --query-vector file://700-rag/test_query_384.json \
    --top-k 3 --return-metadata --return-distance
```

```json
{
    "vectors": [
        {
            "distance": 0.9783104658126831,
            "key": "aaf8fea3-db27-40c4-aa5a-249827c89534",
            "metadata": {
                "product": "HomeAudio 7.1",
                "category": "Speakers",
                "price_range": "$800-$1000",
                "rating": "4.8",
                "source": "electronics.jsonl",
                "text": "The HomeAudio 7.1 system delivers a truly cinematic experience …",
                "combined_text": "Product: HomeAudio 7.1\nCategory: Speakers\nDescription: …"
            }
        },
        {
            "distance": 0.9794089198112488,
            "key": "e2ab430b-10de-4283-8ac3-c6d6f00b99b8",
            "metadata": {"product": "PhotoMaster DSLR", "category": "Cameras", "rating": "4.5", …}
        },
        {
            "distance": 0.980805516242981,
            "key": "159417e0-87d3-406a-93d9-c75d0edcb1c2",
            "metadata": {"product": "GameBox X", "category": "Gaming", "rating": "4.7", …}
        }
    ],
    "distanceMetric": "cosine"
}
```

Three things to read out of this:

- **`"distanceMetric": "cosine"`, distances ~0.978–0.981.** These are cosine *distances*, so
  lower is closer. A synthetic all-`0.1` vector is nearly orthogonal to every real embedding,
  which is exactly why all three cluster at ~0.98 and the ranking is meaningless. This query
  proves the index is queryable, not that retrieval is good.
- **Metadata is returned with the vector.** `product`, `category`, `price_range`, `rating`,
  `source` and the original `text` all ride along. No second lookup into a separate document
  store — the retrieval hop is one API call.
- **Both `text` and `combined_text` are stored.** `combined_text` (`Product: … Category: …
  Description: …`) is what was embedded; `text` is the clean description. Embedding the
  enriched string means a query mentioning a category can match on it, while the prompt gets
  the readable version.

## Deploy the RAG service

```bash
kubectl get endpointslices.discovery.k8s.io -l kubernetes.io/service-name=vllm-serve-svc
```

```
NAME           ADDRESSTYPE   PORTS   ENDPOINTS     AGE
vllm-serve-…   IPv4          8000    10.0.26.208   <age>
```

Confirm vLLM is actually backing the Service before wiring anything to it. A Service with an
empty EndpointSlice resolves fine and connections just hang — check endpoints, not the
Service.

```bash
kubectl apply -f 700-rag/rag-serve.yml

export MODEL_ID="ministral"
export MODEL_ENDPOINT=http://vllm-serve-svc:8000/v1/chat/completions

envsubst < 700-rag/rag-service.yml > 700-rag/rag-service-processed.yml
grep -A 5 -B 5 "S3_VECTOR_BUCKET_NAME" 700-rag/rag-service-processed.yml
kubectl apply -f 700-rag/rag-service-processed.yml
```

```
configmap/rag-serve-script created
            periodSeconds: 10
            failureThreshold: 60
          env:
            - name: ACCOUNT_ID
              value: "<ACCOUNT_ID>"
            - name: S3_VECTOR_BUCKET_NAME
              value: "rag-vectors-<ACCOUNT_ID>-<AWS_REGION>"
            - name: S3_VECTOR_INDEX_NAME
              value: "knowledge-base"
            - name: MODEL_ID
              value: "ministral"
deployment.apps/rag-service created
service/rag-service created
```

`failureThreshold: 60` with `periodSeconds: 10` gives the readiness probe **10 minutes** —
this pod pip-installs `torch` and `sentence-transformers` and downloads the embedding model at
startup. Default probe settings would kill it in a CrashLoopBackOff before it ever came up.
(For production, bake the dependencies and the embedding model into an image instead.)

```bash
kubectl rollout status deployment/rag-service --timeout=300s
kubectl get deployment rag-service
kubectl get pods -l app=rag-service
```

```
Waiting for deployment "rag-service" rollout to finish: 0 of 1 updated replicas are available...
deployment "rag-service" successfully rolled out
NAME          READY   UP-TO-DATE   AVAILABLE   AGE
rag-service   1/1     1            1           <age>
NAME            READY   STATUS    RESTARTS   AGE
rag-service-…   1/1     Running   0          <age>
```

## The A/B that makes the case

`rag-service` exposes `/query` with a `use_rag` flag, so the same service answers both ways.

### `use_rag: true`

```bash
kubectl port-forward svc/rag-service 8080:80

curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the battery life and performance of high-end laptops",
    "use_rag": true
  }' | jq
```

```json
{
  "generated_text": "Based on the provided context, here's a detailed comparison of the **battery life** and **performance** of the **high-end** laptops mentioned (the **PowerBook Pro** and **UltraBook Slim**), while also noting the **BudgetMate 5** for contrast (as it is not a high-end model but included for perspective).\n\n---\n\n### **1. Performance Comparison**\n#### **PowerBook Pro**\n- **Processor & GPU**: Equipped with a **high-wattage processor** and a **dedicated graphics card**, delivering **desktop-class performance**…\n- **Cooling System**: Features an advanced cooling system that keeps temperatures manageable even under **sustained load**, though **fan noise becomes noticeable during intensive tasks**.\n…",
  "query": "Compare the battery life and performance of high-end laptops",
  "rag_enabled": true,
  "vector_store": "S3 Vectors",
  "processing_time": 15.515868902206421,
  "context_used": "[Source: electronics.jsonl - Product: PowerBook Pro - Similarity: 0.567]The PowerBook Pro balances performance and battery life impressively, with its high-wattage processor and dedicated graphics card delivering desktop-class performance while still managing 8-10 hours of mixed use. The cooling system keeps temperatures manageable even under sustained load, though fan noise becomes noticeable during intensive tasks. …[Source: electronics.jsonl - Product: BudgetMate 5 - Similarity: 0.546]The BudgetMate 5 offers impressive value with performance that handles everyday tasks smoothly, though it struggles with demanding applications. Battery life is adequate at 6-7 hours …[Source: electronics.jsonl - Product: UltraBook Slim - Similarity: 0.526]The UltraBook Slim lives up to its name with an incredibly thin profile of just 0.6 inches … Battery life is outstanding, consistently delivering 12-14 hours of real-world use. …"
}
```

Every claim in the answer traces to a retrieved chunk: "8-10 hours of mixed use", "fan noise
becomes noticeable", "12-14 hours". And note what the model did with the *third* result — the
retriever returned BudgetMate 5 at similarity 0.546, but the question asked about *high-end*
laptops, so the model excluded it explicitly ("as it is not a high-end model"). Retrieval is
recall-oriented and imprecise; the generator does the filtering.

`context_used` in the response body is the feature to copy. Shipping the retrieved context
alongside the answer makes RAG debuggable: when an answer is wrong you can immediately tell
whether retrieval failed or generation failed. Those need completely different fixes.

### `use_rag: false`

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the battery life and performance of high-end laptops",
    "use_rag": false
  }' | jq
```

```json
{
  "generated_text": "The **battery life and performance** of **high-end laptops** (e.g., from **Apple, Dell XPS, HP Spectre, Lenovo ThinkPad, Razer Blade, ASUS ROG Zephyrus, MacBook Pro 16\") vary significantly…\n\n| **Laptop** | **Idle (Hours)** | **Real-World (Hours)** | **Notes** |\n|---|---|---|---|\n| **MacBook Pro 16\" (M2 Max)** | ~20+ (Apple TV App) | **8–12 hrs** … |\n| **MacBook Pro 14\" (M3 Max)** | ~20+ | **10–15 hrs** … |\n| **Dell XPS 17 (i9-13",
  "query": "Compare the battery life and performance of high-end laptops",
  "rag_enabled": false,
  "vector_store": "None",
  "processing_time": 15.298224210739136
}
```

Fluent, specific, confidently sourced from nothing. It cites battery figures for named
products from its training data, with no way to verify or update them, and it truncates
mid-table. If your corpus is your own product catalogue, none of this is usable.

### The measurement that surprises people

| | No RAG | With RAG |
|---|---|---|
| Processing time | **15.30 s** | **15.52 s** |
| Grounding | none | 3 sources with similarity scores |
| Auditable | no | yes (`context_used`) |
| Updatable | retrain | re-run the ingest job |

**Retrieval cost 220 ms — about 1.4 % overhead.** The embedding + `QueryVectors` round trip
disappears next to 15 seconds of token generation. This is the number that settles the "is RAG
too slow?" argument: for any generation-heavy workload, vector search is free.

And the update story is the real operational win. New product? Drop a JSONL file in S3 and
re-run the processor Job. No GPU, no training, no redeploy.

## A UI for the comparison

```bash
kubectl apply -f 700-rag/rag-gradio-app.yml
kubectl apply -f 700-rag/rag-gradio-deploy.yml
```

The Gradio app (`700-rag/rag-gradio-app.yml`, a ConfigMap-mounted single Python file) calls
`/query` **twice per submission** — once with `use_rag=True`, once with `False` — and renders
them side by side along with the retrieved context:

```python
API_BASE_URL = "http://rag-service:80"
TOP_K, MAX_TOKENS, TEMPERATURE = 3, 512, 0.7

def query_all_endpoints(query, temperature, max_tokens):
    start_time = time.time()
    rag_results      = make_api_request(query, use_rag=True,  temperature=temperature, max_tokens=max_tokens)
    standard_results = make_api_request(query, use_rag=False, temperature=temperature, max_tokens=max_tokens)
    elapsed_time = time.time() - start_time
    …
```

The Deployment is the same shape as Open WebUI in [Chapter 02](02-serve-with-vllm.md) — a
`python:3.11-slim` container that pip-installs `gradio==5.49.1` at startup, pinned to
`m5.xlarge` by `nodeSelector`, behind an ALB Ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-gradio-alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /
    alb.ingress.kubernetes.io/success-codes: '200-302,307,404'
    alb.ingress.kubernetes.io/load-balancer-name: rag-gradio-alb
    alb.ingress.kubernetes.io/inbound-cidrs: 0.0.0.0/0    # ← see the warning below
spec:
  ingressClassName: alb
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service: {name: rag-gradio-interface, port: {number: 80}}
```

> **`inbound-cidrs: 0.0.0.0/0` overrides the cluster-wide IP lock** from
> [Chapter 00](00-environment-and-cluster.md#lock-the-alb-to-your-ip-before-exposing-anything). The per-Ingress annotation
> beats the `IngressClassParams` default. Change it to `<YOUR_IP>/32` unless you intend to
> publish an unauthenticated LLM endpoint to the internet.

`success-codes: '200-302,307,404'` is also worth noting: Gradio's SPA returns 307s and, before
the app finishes booting, 404s. Without those the target group flaps.

```bash
kubectl wait --for=jsonpath='{.status.loadBalancer.ingress[0].hostname}' \
  ingress/rag-gradio-alb --timeout=300s
export GRADIO_URL=$(kubectl get ingress/rag-gradio-alb \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "http://${GRADIO_URL}"
```

### The app

![RAG Comparison Tool loaded in a browser: query box pre-filled with "Compare the battery life and performance of high-end laptops", Temperature 0.7 and Max Tokens 512 sliders, eight sample-query buttons, and Side-by-Side Comparison / Standard LLM Response / RAG Response tabs with both panels showing "processing | 12.7s"](screenshots/rag-comparison-ui-in-progress.png)

Both requests in flight at 12.7 s, the retrieved-context panel at 14.8 s. The eight sample
queries span the corpus (laptops, smart lighting, voice recognition, privacy, cooling), which
matters for a demo — a query with nothing relevant in 25 records makes RAG look broken when
it's the corpus that's empty.

![The same tool after submission: "Query completed in 30.77 seconds". Left panel "LLM Response (No RAG)" lists Apple/Dell/HP/Lenovo/Razer/ASUS and a table row for MacBook Pro 14 (M4 Pro) with 48-core GPU, processing time 15.31s. Right panel "RAG-Enhanced Response" compares PowerBook Pro and UltraBook Slim from the retrieved context, explicitly excluding BudgetMate 5, processing time 15.46s. The Retrieved Context panel below shows the three sources with similarities 0.567, 0.546 and 0.526](screenshots/rag-comparison-ui-results.png)

Side by side, the difference is unmissable:

| | LLM (No RAG) — 15.31 s | RAG-Enhanced — 15.46 s |
|---|---|---|
| Subjects | Apple, Dell XPS, HP Spectre, ThinkPad, Razer Blade, ROG Zephyrus | PowerBook Pro, UltraBook Slim (from the corpus) |
| Specificity | "MacBook Pro 14" (M4 Pro) \| 48-core GPU (13.2B transistors)" | "high-wattage processor … desktop-class components" |
| Verifiable | **no** | yes — see Retrieved Context |
| Reasoning shown | none | excludes BudgetMate 5 as not high-end |

That "48-core GPU, 13.2B transistors" cell is the whole lesson. It is precise, plausible,
formatted as a data table — and unverifiable. **RAG doesn't make the model more articulate; it
makes the model's claims checkable.** For an internal knowledge base, a support assistant, or
anything a customer sees, that's the only property that matters.

`Query completed in 30.77 seconds` for the pair, i.e. ~15 s each, confirming from the browser
what the curls showed: retrieval overhead is noise.

## Take-aways

1. **S3 Vectors removes the vector-database from your architecture.** A bucket, an index, two
   API calls. No cluster to size, patch, or pay for while idle.
2. **Embed on CPU, generate on GPU.** `all-MiniLM-L6-v2` at 384 dims on `torch+cpu` costs
   almost nothing. Only generation needs the L40S.
3. **Return the context with the answer.** `context_used` turns an opaque failure into a
   two-way diagnosis: bad retrieval or bad generation.
4. **Retrieval overhead is ~1 %.** 220 ms against 15 s of decode.
5. **Store rich metadata in the index.** Product, category, price, rating and source came back
   in the same call — enough to filter, cite, and render without touching another datastore.
6. **RAG and fine-tuning are orthogonal.** [Chapter 07](07-lora-fine-tuning.md) fixed the
   *form* of the answer; this chapter fixed the *facts*. Nothing stops you from doing both:
   serve the LoRA adapter and inject retrieved context into its prompt.

## Clean up

```bash
kubectl delete deployment rag-gradio-interface 2>/dev/null || true
kubectl delete service rag-gradio-interface 2>/dev/null || true
kubectl delete ingress rag-gradio-alb 2>/dev/null || true
kubectl delete deployment rag-service 2>/dev/null || true
kubectl delete service rag-service 2>/dev/null || true
kubectl delete job rag-document-processor 2>/dev/null || true

kubectl delete deployment mistral 2>/dev/null || true
kubectl delete deployment anyvc 2>/dev/null || true
kubectl delete service vllm-serve-svc 2>/dev/null || true
kubectl delete ingress open-webui-ingress 2>/dev/null || true
kubectl delete deployment open-webui 2>/dev/null || true
kubectl delete service open-webui 2>/dev/null || true
```

```
deployment.apps "anyvc" deleted from default namespace
service "vllm-serve-svc" deleted from default namespace
ingress.networking.k8s.io "open-webui-ingress" deleted from default namespace
deployment.apps "open-webui" deleted from default namespace
service "open-webui" deleted from default namespace
```

The GPU must be free for Chapter 09, which brings up its own vLLM under Ray.

Next: [09 — Ray Serve with KubeRay](09-ray-serve.md)
