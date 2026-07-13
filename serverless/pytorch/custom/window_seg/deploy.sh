#!/bin/bash
# Deploy window_seg model as a Nuclio function
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NUCLIO_DIR="$SCRIPT_DIR/nuclio"

# Check if GPU variant should be used
USE_GPU="${1:-cpu}"

if [ "$USE_GPU" = "gpu" ]; then
    FUNC_CONFIG="$NUCLIO_DIR/function-gpu.yaml"
else
    FUNC_CONFIG="$NUCLIO_DIR/function.yaml"
fi

echo "Deploying window_seg function ($USE_GPU)..."
echo "Function config: $FUNC_CONFIG"
echo "Function path: $NUCLIO_DIR"

# Check that the checkpoint exists in the nuclio dir
if [ ! -f "$NUCLIO_DIR/window_seg_best.pth" ]; then
    echo "ERROR: window_seg_best.pth not found in $NUCLIO_DIR"
    echo "Please copy your checkpoint there first."
    exit 1
fi

if [ ! -f "$NUCLIO_DIR/window_cls_best.pth" ]; then
    echo "ERROR: window_cls_best.pth not found in $NUCLIO_DIR"
    echo "Please copy your classifier checkpoint there first."
    exit 1
fi

# Create project if it doesn't exist
nuctl create project cvat --platform local 2>/dev/null || true

nuctl deploy --project-name cvat --path "$NUCLIO_DIR" \
    --file "$FUNC_CONFIG" --platform local \
    --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
    --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

echo ""
echo "Deployment complete!"
nuctl get function --platform local
