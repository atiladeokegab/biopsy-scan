#!/usr/bin/env bash
# Deploy the standalone usage monitor (monitor/) to its own Cloud Run service: scripts/deploy-gcp-monitor.sh [git-ref, default origin/integration]
# Same as scripts/deploy-gcp.sh (same Dockerfile and nginx.conf), but the site root is monitor/. Deploys the committed ref.
set -euo pipefail
REF="${1:-origin/integration}"
PROJECT="${GCP_PROJECT:-hackathon-starter-ao}"
REGION="${GCP_REGION:-europe-west2}"
SERVICE="${GCP_SERVICE:-preflight-monitor}"
export PATH="/snap/bin:$PATH"
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git archive "$REF" | tar -x -C "$tmp/" && mkdir "$tmp/ctx"
(cd "$tmp" && python3 -I monitor/build.py --check)
cp deploy/gcp/Dockerfile deploy/gcp/nginx.conf "$tmp/ctx/" && cp -r "$tmp/monitor" "$tmp/ctx/site" && rm -f "$tmp/ctx/site/build.py"
gcloud run deploy "$SERVICE" --source "$tmp/ctx" --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated --port 8080 --cpu 1 --memory 128Mi --min-instances 1 --max-instances 3 --quiet
url="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')"
for p in / data.json; do printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "$url/$p")" "$url/$p"; done
echo "deployed $(git rev-parse --short "$REF") to $url"
