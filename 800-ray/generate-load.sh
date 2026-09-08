#!/usr/bin/env bash
#
# Generates sustained concurrent inference load against the Ray Serve deployment so the
# queueing and concurrency panels on the Ray Serve Inference Overview dashboard have
# something to show.
#
# Why this is needed, and why chatting in Open WebUI is not enough:
#
#   The vLLM engine decodes at most MAX_NUM_SEQS (4) sequences at a time, and Ray Serve
#   sends at most max_ongoing_requests (5) to a replica. Nothing queues until more than
#   that arrive at once. A person typing in a chat window never has more than one request
#   in flight, so the queueing panels stay flat at zero no matter how many messages
#   they send.
#
#   The load also has to be sustained rather than a single burst. Queue depth and batch
#   occupancy are instantaneous gauges, and a burst that starts and finishes between two
#   scrapes is invisible. Sixty seconds is the practical minimum; the default below is
#   longer so the pattern is unmistakable on the graphs.
#
# Prerequisite: forward the Ray Serve port in a separate terminal and leave it running:
#
#   kubectl port-forward svc/vllm-serve-svc 8000:8000
#
# Usage:
#   bash 800-ray/generate-load.sh [DURATION_SECONDS] [CONCURRENCY]
#
# Defaults to 180 seconds at 8 concurrent requests, which is comfortably above both the
# engine batch cap and the Serve per-replica cap.

set -euo pipefail

DURATION="${1:-180}"
CONCURRENCY="${2:-8}"
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1/chat/completions}"
MODEL="${MODEL:-ministral}"

PROMPT="Explain how Kubernetes horizontal pod autoscaling works, then walk through what happens step by step when CPU usage spikes on a deployment with three replicas. Aim for about 400 words."

if ! curl --silent --fail --max-time 10 --output /dev/null "http://localhost:8000/v1/models"; then
  echo "ERROR: cannot reach the model at http://localhost:8000"
  echo "Start the port-forward in another terminal first:"
  echo "  kubectl port-forward svc/vllm-serve-svc 8000:8000"
  exit 1
fi

echo "Generating load for ${DURATION}s at ${CONCURRENCY} concurrent requests."
echo "Watch the 'Queueing and concurrency' row of the Ray Serve Inference Overview dashboard."
echo

sent=0
deadline=$(( SECONDS + DURATION ))

while [ "${SECONDS}" -lt "${deadline}" ]; do
  for _ in $(seq 1 "${CONCURRENCY}"); do
    curl --silent --output /dev/null --max-time 120 \
      "${ENDPOINT}" \
      --header 'Content-Type: application/json' \
      --data "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":600}" &
  done
  wait
  sent=$(( sent + CONCURRENCY ))
  echo "  sent ${sent} requests (${SECONDS}s elapsed)"
done

echo
echo "Done. Sent approximately ${sent} requests."
echo "Allow up to 30 seconds for the final scrape, then review the dashboard."
