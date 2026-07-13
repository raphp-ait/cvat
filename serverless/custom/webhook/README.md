# CVAT Webhook Service

Webhook receiver + task processor for automatic window-segmentation mask upload to CVAT.

## What It Does

- Accepts CVAT webhook events at `/webhook/task-created`
- Processes only `create:task` events
- Ignores update events to avoid duplicate processing
- Runs segmentation on task frames and uploads mask annotations
- Supports manual processing endpoint at `/webhook/process`

## Prerequisites

- Docker with the `docker compose` plugin
- Reachable CVAT instance
- Window segmentation checkpoint file (default local path: `./window_seg_best.pth`)

## Quick Start

```bash
cd /mnt/bigDisk/home/picklr/CVAT/cvat/serverless/custom/webhook
cp .env.example .env
# edit .env and set credentials
./run.sh start
```

Health check:

```bash
curl http://localhost:5000/health
```

## Configuration

Edit `.env`:

```env
CVAT_URL=http://localhost:8080
CVAT_USERNAME=admin
CVAT_PASSWORD=your_password_here
WEBHOOK_SECRET=your_secret_key_here
WINDOW_SEG_CHECKPOINT_HOST=./window_seg_best.pth
SEGMENTATION_THRESHOLD=0.5
```

## Endpoints

### Webhook Endpoint

```bash
curl -X POST http://localhost:5000/webhook/task-created \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your_secret_key_here" \
  -d '{"event":"create:task","task":{"id":123,"name":"My task"}}'
```

Non-create events are ignored by design:

```bash
curl -X POST http://localhost:5000/webhook/task-created \
  -H "Content-Type: application/json" \
  -d '{"event":"update:task","task":{"id":123,"name":"My task"}}'
```

### Manual Processing

```bash
curl -X POST http://localhost:5000/webhook/process \
  -H "Content-Type: application/json" \
  -d '{"task_id":123,"async":true}'
```

### Status

```bash
curl http://localhost:5000/status/123
```

## Operations

```bash
./run.sh start
./run.sh stop
./run.sh restart
./run.sh build
./run.sh logs
./run.sh shell
./run.sh standalone
```

## Notes

- The service uses host networking in `docker-compose.yml`.
- With host networking, `CVAT_URL` should point to an address reachable from the host.
- The runtime image includes only production files (`app.py`, `window_seg_inference.py`).
