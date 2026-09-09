# 04 — An agentic app with the Strands Agents SDK

Goal: run a tool-calling agent inside the cluster whose "LLM provider" is the vLLM Service
from Chapter 02. No external model API, no egress to a hosted provider, no API keys.

This is the chapter that justifies the previous three. Self-hosting a model is only
interesting if applications can consume it as easily as they'd consume a SaaS endpoint.

> **In plain terms — what an "agent" actually is.** There's no magic here, and the mechanism is
> worth spelling out because the word gets used loosely.
>
> A language model can only produce text. It cannot check the weather. So you tell it, in the
> request, *"here is a list of functions you may call, with their parameters."* When the model
> decides it needs one, it doesn't answer the user — it emits a structured request like
> `get_weather(city="Denver")`. **Your code** runs that function. You then send the result back
> to the model as another message, and it writes the human answer.
>
> ```
> user: "what's the weather in Denver?"
>   → model: [call get_weather(city="Denver")]      ← model stops and asks for a tool
>   → your code calls a weather API                 ← the model never touches the network
>   → model: "It's clear and 30 °C in Denver."      ← model turns data into prose
> ```
>
> That loop is the whole idea. An "agent framework" like **Strands** just manages it for you:
> describing the tools, parsing the call, running it, feeding the result back, and repeating
> until the model produces a final answer instead of another tool call.
>
> The part that should genuinely surprise you is that this works on an **8-billion-parameter
> model running on one GPU you control.** Reliable tool calling used to require a frontier hosted
> model. Both demos below — a timezone lookup and a live weather call — are real output from the
> Chapter 02 server.
>
> And note what makes it possible with no adapter code at all: vLLM speaks the OpenAI API, so
> Strands points at an in-cluster DNS name (`vllm-serve-svc:8000`) instead of a vendor URL.
> No egress, no API key, no per-token bill.

## Build and push the agent image

```bash
export ECR_REPO=$(aws ecr describe-repositories \
  --repository-names strands-weather-agent \
  --query 'repositories[0].repositoryUri' --output text)
echo "Your ECR repository URI is: $ECR_REPO"
```

```
Your ECR repository URI is: <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-weather-agent
```

```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $(echo $ECR_REPO | cut -d'/' -f1)

docker build -t strands-weather-agent 200-strands-agent
docker tag strands-weather-agent:latest $ECR_REPO:latest
docker push $ECR_REPO:latest
```

```
Login Succeeded
[+] Building 29.0s (10/10) FINISHED
 => [1/5] FROM docker.io/library/python:3.12-slim         2.7s
 => [2/5] WORKDIR /app                                    0.2s
 => [3/5] COPY requirements.txt .                         0.0s
 => [4/5] RUN pip install -r requirements.txt            23.0s
 => [5/5] COPY strands-agent.py app.py                    0.0s
latest: digest: sha256:6d2df7d5dff4f01091a5ba1fe3407883faa7b4fe3ab44bb099d5956cdb963acf size: 1993
```

29 seconds, and 23 of those are `pip install`. The whole agent is a `python:3.12-slim` base,
a requirements file, and **one** application file (`strands-agent.py` → `app.py`). Contrast
with the multi-GB vLLM image: agents are cheap, models are expensive. That asymmetry is the
reason you centralize the model behind a Service and let many small agents share it.

## Deploy it

`200-strands-agent/vllm-deployment-agents.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-weather-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: strands-weather-agent
  template:
    metadata:
      labels:
        app: strands-weather-agent
    spec:
      containers:
      - name: strands-weather-agent
        image: $ECR_REPO:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_ENDPOINT
          value: "http://vllm-serve-svc:8000/v1"   # ← in-cluster, OpenAI-compatible
        - name: MODEL_ID
          value: "ministral"                       # ← --served-model-name from Chapter 02
---
apiVersion: v1
kind: Service
metadata:
  name: strands-weather-agent
spec:
  selector:
    app: strands-weather-agent
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

```bash
export ECR_REPO=$(aws ecr describe-repositories --repository-names strands-weather-agent \
  --query 'repositories[0].repositoryUri' --output text)

envsubst < 200-strands-agent/vllm-deployment-agents.yaml | kubectl apply -f -
kubectl wait pods --for=jsonpath='{.status.phase}'=Running \
  -l app=strands-weather-agent --timeout=300s
```

```
deployment.apps/strands-weather-agent created
service/strands-weather-agent created
pod/strands-weather-agent-… condition met
```

The entire model configuration is two environment variables. `MODEL_ENDPOINT` +
`MODEL_ID` — the same two values you'd point at OpenAI, aimed at a ClusterIP instead. Note
what's *absent*: no API key, no secret, no NAT gateway egress, no per-token bill. Traffic
never leaves the VPC.

There's also no `nodeSelector` here. The agent is a thin HTTP client; Karpenter puts it on a
CPU node and it stays there.

## Exercise the tools

```bash
kubectl port-forward svc/strands-weather-agent 8080:80
```

```bash
curl -s -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What time is it in Dallas?"}' \
  | jq -r '.response.message.content[0].text'
```

```
The current time in Dallas is **2:54 PM CDT**.
```

```bash
curl -s -X POST http://localhost:8080/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Denver?"}' \
  | jq -r '.response.message.content[0].text'
```

```
The current weather in **Denver** is **clear sky** with a temperature of **30.0°C**.
```

Both answers are facts the model cannot know — the current time and live weather. Getting
them right proves the full agent loop executed against a self-hosted model:

1. The agent sent the user query **plus tool schemas** to `/v1/chat/completions`.
2. The 8 B model emitted a structured tool call, correctly extracting `Dallas` / `Denver` as
   arguments.
3. The Strands runtime executed the tool and fed the result back.
4. The model composed a natural-language answer from the tool output.

Two things worth flagging for anyone planning to build on this:

- **Tool calling works on a small self-hosted model.** vLLM's startup log said
  `"auto" tool choice has been enabled` — it parses the model's tool-call syntax and returns
  OpenAI-shaped `tool_calls`. You don't need a frontier model to get reliable function
  calling for narrow, well-described tools.
- **The response envelope is `.response.message.content[0].text`** — Strands' own structure,
  not OpenAI's. The SDK owns the agent loop; vLLM only sees chat completions.

## Clean up before the next chapter

```bash
kubectl delete service strands-weather-agent -n default
kubectl delete deployment strands-weather-agent -n default
```

```
service "strands-weather-agent" deleted from default namespace
deployment.apps "strands-weather-agent" deleted from default namespace
```

Leave vLLM running — Chapter 05 benchmarks it, and you want a clean engine with no competing
traffic.

Next: [05 — Benchmarking](05-benchmarking.md)
