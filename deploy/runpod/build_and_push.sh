#!/usr/bin/env bash
# build_and_push.sh — Build and push NorthStar Navigator Docker image
#
# Usage:
#   ./deploy/runpod/build_and_push.sh v0.1.0       # explicit version
#   ./deploy/runpod/build_and_push.sh --dry-run v0.1.0  # build only
#   ./deploy/runpod/build_and_push.sh --help

set -euo pipefail

REPO="wanderduck/northstar-navigator"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
Build and push NorthStar Navigator Docker image to Docker Hub.

Usage:
    ./deploy/runpod/build_and_push.sh v0.1.0           # build + push
    ./deploy/runpod/build_and_push.sh --dry-run v0.1.0 # build only
    ./deploy/runpod/build_and_push.sh --help

Always builds with --platform=linux/amd64 for RunPod.
USAGE
    exit 0
fi

DRY_RUN=false
VERSION=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        v*) VERSION="$arg" ;;
        *) echo "Unknown: $arg (try --help)"; exit 1 ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    echo "ERROR: Version required (e.g., v0.1.0)"
    exit 1
fi

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Version must match vX.Y.Z (got: $VERSION)"
    exit 1
fi

TAG="$REPO:$VERSION"
echo "=== Building $TAG ==="
echo "    Platform: linux/amd64"

docker build \
    --platform=linux/amd64 \
    --tag "$TAG" \
    --file "$SCRIPT_DIR/Dockerfile" \
    "$PROJECT_ROOT"

echo "=== Build complete: $TAG ==="

if [[ "$DRY_RUN" == true ]]; then
    echo "=== Dry run — skipping push ==="
    exit 0
fi

echo "=== Pushing $TAG ==="
docker push "$TAG"
echo "=== Done: docker.io/$TAG ==="
