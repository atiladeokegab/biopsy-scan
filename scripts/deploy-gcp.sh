#!/usr/bin/env bash
# Redeploy the static demo/ to Cloud Run. Run after every merge: scripts/deploy-gcp.sh [git-ref, default origin/integration]
# Deploys the committed ref, not your working tree, and runs the same data steps as .github/workflows/pages.yml.
set -euo pipefail
REF="${1:-origin/integration}"
PROJECT="${GCP_PROJECT:-hackathon-starter-ao}"
REGION="${GCP_REGION:-europe-west2}"
SERVICE="${GCP_SERVICE:-preflight-demo}"
export PATH="/snap/bin:$PATH"
cd "$(git rev-parse --show-toplevel)"
git fetch -q origin
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git archive "$REF" | tar -x -C "$tmp/" && mkdir "$tmp/ctx"
(cd "$tmp" && python3 -I demo/collect_race.py && python3 -I demo/check_data.py demo/data && python3 -I demo/collect_preflight.py --check)
cp deploy/gcp/Dockerfile deploy/gcp/nginx.conf "$tmp/ctx/" && cp -r "$tmp/demo" "$tmp/ctx/site"
gcloud run deploy "$SERVICE" --source "$tmp/ctx" --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated --port 8080 --cpu 1 --memory 128Mi --min-instances 1 --max-instances 3 --quiet
url="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')"
for p in / app/ plan/ usage/; do printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "$url/$p")" "$url/$p"; done
echo "deployed $(git rev-parse --short "$REF") to $url"
