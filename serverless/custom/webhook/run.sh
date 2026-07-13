#!/bin/bash
# Run the CVAT webhook service
set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

echo "========================================="
echo "CVAT Webhook Service"
echo "========================================="
echo ""

MODE="${1:-start}"
COMPOSE_CMD="docker compose"

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose plugin is required"
    exit 1
fi

case "$MODE" in
    start)
        echo "Starting webhook service in Docker..."
        
        # Check for .env file
        if [ ! -f ".env" ]; then
            echo "Creating .env configuration file..."
            cat > .env << 'EOF'
# CVAT Configuration
CVAT_URL=http://localhost:8080
CVAT_USERNAME=admin
CVAT_PASSWORD=your_password_here

# Webhook Configuration
WEBHOOK_SECRET=your_secret_key_here
WEBHOOK_PORT=5000

# Model Configuration
WINDOW_SEG_CHECKPOINT_HOST=./window_seg_best.pth
SEGMENTATION_THRESHOLD=0.5
EOF
            echo "✓ Created .env file"
            echo "IMPORTANT: Edit .env with your actual CVAT credentials"
            echo ""
            exit 1
        fi

        # Validate required variables without sourcing .env.
        if ! grep -q '^CVAT_PASSWORD=' .env || grep -q '^CVAT_PASSWORD=your_password_here$' .env; then
            echo "ERROR: Please set CVAT_PASSWORD in .env file"
            exit 1
        fi

        # Start the service
        $COMPOSE_CMD up -d --build
        
        echo ""
        echo "✓ Webhook service started!"
        echo ""
        echo "Commands:"
        echo "  ./run.sh logs    - View logs"
        echo "  ./run.sh stop    - Stop service"
        echo "  ./run.sh restart - Restart service"
        echo ""
        echo "Health check: curl http://localhost:5000/health"
        echo ""
        ;;
    
    stop)
        echo "Stopping webhook service..."
        $COMPOSE_CMD down
        echo "✓ Stopped"
        ;;
    
    restart)
        echo "Recreating webhook service with latest image..."
        $COMPOSE_CMD up -d --build --force-recreate
        echo "✓ Recreated"
        ;;
    
    logs)
        echo "Webhook service logs:"
        echo ""
        docker logs -f cvat-webhook
        ;;
    
    build)
        echo "Building Docker image..."
        $COMPOSE_CMD build --pull
        echo "✓ Build complete"
        ;;
    
    shell)
        echo "Opening shell in container..."
        $COMPOSE_CMD exec webhook bash
        ;;
    
    standalone)
        echo "Running in standalone mode (no Docker)..."
        
        # Check environment variables
        if [ -z "$CVAT_USERNAME" ] || [ -z "$CVAT_PASSWORD" ]; then
            echo "❌ ERROR: CVAT_USERNAME and CVAT_PASSWORD required"
            echo ""
            echo "Set them first:"
            echo "  export CVAT_USERNAME=admin"
            echo "  export CVAT_PASSWORD=your_password"
            echo ""
            exit 1
        fi
        
        # Check dependencies
        if ! python -c "import flask" 2>/dev/null; then
            echo "Installing Python dependencies..."
            pip install -r requirements.txt
            echo ""
        fi
        
        # Check checkpoint
        CHECKPOINT="${WINDOW_SEG_CHECKPOINT:-./window_seg_best.pth}"
        if [ ! -f "$CHECKPOINT" ]; then
            echo "❌ ERROR: Model checkpoint not found: $CHECKPOINT"
            exit 1
        fi
        
        # Run the app
        python app.py --host 0.0.0.0 --port 5000 ${DEBUG:+--debug}
        ;;
    
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start      - Start service in Docker (default)"
        echo "  stop       - Stop service"
        echo "  restart    - Restart service"
        echo "  logs       - View logs"
        echo "  build      - Build Docker image"
        echo "  shell      - Open shell in container"
        echo "  standalone - Run without Docker"
        echo ""
        exit 1
        ;;
esac
