#!/usr/bin/env python3
"""
CVAT Webhook Receiver and Task Processor

This single script handles both:
1. Webhook reception (Flask server)
2. Task processing (window segmentation)

Architecture:
- Flask receives webhook requests
- Spawns background threads for task processing
- Uses CVAT SDK to download images, process, and upload annotations

Usage:
    python app.py --host 0.0.0.0 --port 5000
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify
from PIL import Image

from window_seg_inference import WindowSegInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# Configuration
# ============================================================================

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "cvat_webhook_secret")
CVAT_URL = os.getenv("CVAT_URL", "http://ls1hvs03:8080")
CVAT_USERNAME = os.getenv("CVAT_USERNAME", "admin")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD", "Changeme!")
CONFIDENCE_THRESHOLD = float(os.getenv("SEGMENTATION_THRESHOLD", "0.5"))
JOB_WAIT_TIMEOUT_SECONDS = int(os.getenv("JOB_WAIT_TIMEOUT_SECONDS", "10"))
JOB_WAIT_POLL_INTERVAL_SECONDS = float(os.getenv("JOB_WAIT_POLL_INTERVAL_SECONDS", "3"))


# ============================================================================
# Model Inference
# ============================================================================

def _build_inference_model() -> WindowSegInference:
    checkpoint_path = os.getenv("WINDOW_SEG_CHECKPOINT", "/app/window_seg_best.pth")
    model = WindowSegInference(
        checkpoint_path=checkpoint_path,
        threshold=CONFIDENCE_THRESHOLD,
    )
    logger.info(f"Window segmentation model loaded from: {model.checkpoint_path}")
    return model


# ============================================================================
# Task Processor
# ============================================================================

class TaskProcessor:
    """Process CVAT tasks with automatic window segmentation annotations."""
    
    def __init__(self, cvat_url, username, password, model):
        """
        Initialize the processor.
        
        Args:
            cvat_url: CVAT server URL
            username: CVAT username
            password: CVAT password
            model: WindowSegInference instance
        """
        self.cvat_url = cvat_url
        self.username = username
        self.password = password
        self.model = model
        
        # Initialize CVAT SDK using low-level API (official approach)
        from cvat_sdk.api_client import ApiClient, Configuration
        from cvat_sdk.api_client.api import tasks_api, labels_api, jobs_api
        
        configuration = Configuration(
            host=cvat_url,
            username=username,
            password=password,
        )
        
        self.api_client = ApiClient(configuration)
        self.tasks_api = tasks_api.TasksApi(self.api_client)
        self.labels_api = labels_api.LabelsApi(self.api_client)
        self.jobs_api = jobs_api.JobsApi(self.api_client)
        
        logger.info("CVAT client initialized (low-level API)")

    def _wait_for_jobs(self, task_id):
        """Wait briefly for CVAT to create jobs after task creation."""
        deadline = time.time() + max(0, JOB_WAIT_TIMEOUT_SECONDS)
        attempt = 0

        while True:
            attempt += 1
            jobs_response, _ = self.jobs_api.list(task_id=task_id)
            if jobs_response and jobs_response.results:
                if attempt > 1:
                    logger.info(
                        f"Jobs became available for task {task_id} after {attempt} poll(s)"
                    )
                return jobs_response.results

            if time.time() >= deadline:
                return []

            logger.info(
                f"No jobs for task {task_id} yet; waiting {JOB_WAIT_POLL_INTERVAL_SECONDS:.1f}s before retry"
            )
            time.sleep(max(0.1, JOB_WAIT_POLL_INTERVAL_SECONDS))
    
    def process_task(self, task_id):
        """
        Process all images in a CVAT task.
        
        Args:
            task_id: ID of the CVAT task to process
            
        Returns:
            dict: Processing result
        """
        logger.info(f"Processing task {task_id}")
        
        try:
            # Get task details using low-level API
            task, _ = self.tasks_api.retrieve(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return {"status": "failed", "error": f"Task {task_id} not found"}
            
            logger.info(f"Task: {task.name}, Status: {task.status}")
            
            # Get or create label
            label = self._get_or_create_label(task_id)
            if not label:
                return {"status": "failed", "error": "Could not get or create label"}
            
            # Get jobs for this task
            jobs = self._wait_for_jobs(task_id)
            if not jobs:
                logger.warning(
                    f"No jobs found for task {task_id} after waiting {JOB_WAIT_TIMEOUT_SECONDS}s"
                )
                return {"status": "completed", "task_id": task_id, "processed_frames": 0, "total_frames": 0}
            
            # Process each job
            total_frames = 0
            processed_frames = 0
            
            for job in jobs:
                logger.info(f"Processing job {job.id}")
                
                # Get frames for this job using retrieve_data_meta
                try:
                    meta_response, _ = self.jobs_api.retrieve_data_meta(id=job.id)
                    if not meta_response or not hasattr(meta_response, 'frames') or not meta_response.frames:
                        logger.warning(f"No frames found for job {job.id}")
                        continue
                    
                    for frame_offset, _ in enumerate(meta_response.frames):
                        frame_number = meta_response.start_frame + frame_offset
                        total_frames += 1
                        logger.info(f"  Frame {frame_number}")
                        
                        try:
                            pil_image = self._get_image(job.id, frame_number)
                            annotations = self._process_image(pil_image, frame_number, label.id)

                            if annotations:
                                self._upload_job_annotations(job.id, annotations)
                                processed_frames += 1
                                logger.info(f"    ✓ Uploaded {len(annotations)} annotation(s)")
                            else:
                                logger.info(f"    - No windows detected")
                        
                        except Exception as e:
                            logger.error(f"    ✗ Error processing frame: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {e}")
                    continue
            
            logger.info(f"Task {task_id} complete! Processed {processed_frames}/{total_frames} frames")
            
            return {
                "status": "completed",
                "task_id": task_id,
                "processed_frames": processed_frames,
                "total_frames": total_frames
            }
        
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            return {"status": "failed", "task_id": task_id, "error": str(e)}
    
    def _get_image(self, job_id, frame_number):
        """Retrieve an individual frame image from a CVAT job."""
        image_data, _ = self.jobs_api.retrieve_data(
            job_id,
            type="frame",
            number=frame_number,
            quality="original",
        )

        image_data.seek(0)
        with Image.open(image_data) as image:
            return image.copy()

    def _process_image(self, image, frame_number, label_id):
        """Run model inference and return CVAT shape payloads for one frame."""
        model_results = self.model.segment_image(image)

        annotations = []
        for result in model_results:
            if "mask" not in result:
                continue

            annotations.append(
                {
                    "type": "mask",
                    "frame": frame_number,
                    "label_id": label_id,
                    "points": [float(v) for v in result["mask"]],
                    "group": 0,
                    "source": "auto",
                    "occluded": False,
                    "outside": False,
                    "z_order": 0,
                    "rotation": 0,
                    "attributes": [],
                }
            )

        return annotations
    
    def _get_or_create_label(self, task_id):
        """Get a label for annotations from labels defined on the task."""
        try:
            labels_response, _ = self.labels_api.list(task_id=task_id, page_size=100)
            task_labels = labels_response.results if labels_response and labels_response.results else []

            if not task_labels:
                logger.error(f"No labels found on task {task_id}. Cannot upload annotations.")
                return None

            requested_label_id = os.getenv("WINDOW_SEG_LABEL_ID")
            if requested_label_id:
                try:
                    requested_label_id = int(requested_label_id)
                except ValueError:
                    logger.warning(f"Invalid WINDOW_SEG_LABEL_ID={requested_label_id}; using first task label")
                else:
                    for label in task_labels:
                        if label.id == requested_label_id:
                            logger.info(f"Using task label from WINDOW_SEG_LABEL_ID: {label.name} (ID: {label.id})")
                            return label
                    logger.warning(f"WINDOW_SEG_LABEL_ID={requested_label_id} not found on task {task_id}; using first task label")

            selected_label = task_labels[0]
            logger.info(f"Using first task label: {selected_label.name} (ID: {selected_label.id})")
            return selected_label
        except Exception as e:
            logger.error(f"Failed to load task labels from CVAT for task {task_id}: {e}")
            return None
    
    def _upload_job_annotations(self, job_id, annotations):
        """Upload segmentation annotations to a job."""
        from cvat_sdk.api_client.models import LabeledShapeRequest, ShapeType, PatchedLabeledDataRequest
        
        # Prepare the request
        shapes = []
        for ann in annotations:
            shape_type = ann.get('type', 'mask')
            if isinstance(shape_type, str):
                shape_type = ShapeType(shape_type)

            shape = LabeledShapeRequest(
                type=shape_type,
                frame=ann['frame'],
                label_id=ann['label_id'],
                points=ann['points'],
                group=ann.get('group', 0),
                source=ann.get('source', "auto"),
                occluded=ann.get('occluded', False),
                outside=ann.get('outside', False),
                z_order=ann.get('z_order', 0),
                rotation=ann.get('rotation', 0),
                attributes=ann.get('attributes', []),
            )
            shapes.append(shape)
        
        if not shapes:
            return
        
        request = PatchedLabeledDataRequest(shapes=shapes)
        
        # Upload annotations using low-level API
        try:
            self.jobs_api.partial_update_annotations(
                id=job_id,
                action="create",
                patched_labeled_data_request=request
            )
            logger.info(f"Uploaded {len(shapes)} annotations to job {job_id}")
        except Exception as e:
            logger.error(f"Failed to upload annotations to job {job_id}: {e}")
            raise


# ============================================================================
# Background Task Manager
# ============================================================================

class BackgroundTaskManager:
    """Manage background task processing."""
    
    def __init__(self, processor):
        self.processor = processor
        self.active_tasks = {}
        self.lock = threading.Lock()
    
    def process_async(self, task_id):
        """Process a task in a background thread."""
        with self.lock:
            existing = self.active_tasks.get(task_id)
            if existing and existing.get("status") in {"processing", "completed"}:
                logger.info(
                    f"Skipping duplicate async processing request for task {task_id} (status={existing.get('status')})"
                )
                return {
                    "status": existing.get("status"),
                    "task_id": task_id,
                    "message": "Task already processing or completed",
                }

            self.active_tasks[task_id] = {
                "status": "processing",
                "task_id": task_id,
            }

        def run_processing():
            result = self.processor.process_task(task_id)
            
            with self.lock:
                self.active_tasks[task_id] = result
        
        thread = threading.Thread(target=run_processing, daemon=True)
        thread.start()
        
        return {
            "status": "processing",
            "task_id": task_id,
            "message": "Task processing started in background"
        }
    
    def get_status(self, task_id):
        """Get processing status for a task."""
        with self.lock:
            return self.active_tasks.get(task_id, {"status": "unknown"})


# ============================================================================
# Flask Routes
# ============================================================================

# Global instances (initialized on first request)
_model = None
_processor = None
_task_manager = None


def get_model():
    """Lazy initialization of the window segmentation model."""
    global _model
    if _model is None:
        _model = _build_inference_model()
    return _model


def get_processor():
    """Lazy initialization of the processor."""
    global _processor
    if _processor is None:
        _processor = TaskProcessor(CVAT_URL, CVAT_USERNAME, CVAT_PASSWORD, get_model())
    return _processor


def get_task_manager():
    """Lazy initialization of the task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager(get_processor())
    return _task_manager


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "cvat-webhook-receiver",
        "cvat_url": CVAT_URL,
        "mode": "MODEL (automatic window segmentation)"
    })


@app.route('/webhook/task-created', methods=['POST'])
def handle_task_created():
    """Handle task creation webhook from CVAT."""
    logger.info("Received task creation webhook")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request data: {request.get_data(as_text=True)}")
    
    # Verify webhook secret if provided
    secret = request.headers.get('X-Webhook-Secret')
    if secret and secret != WEBHOOK_SECRET:
        logger.warning("Invalid webhook secret")
        return jsonify({"error": "Invalid webhook secret"}), 401
    
    # Parse request body
    try:
        data = request.get_json()
        logger.info(f"Parsed JSON data: {data}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # CVAT sends task info nested in 'task' object
        task_data = data.get('task', {})
        task_id = task_data.get('id')
        
        if not task_id:
            logger.error(f"task_id not found. Data structure: {data.keys()}")
            return jsonify({"error": "task_id is required"}), 400
        
        task_name = task_data.get('name', 'Unknown')
        event_type = data.get('event', 'unknown')
        logger.info(f"Event: {event_type}, Task ID={task_id}, Name={task_name}")

        if event_type != 'create:task':
            logger.info(
                f"Ignoring webhook event '{event_type}' for task {task_id}; only 'create:task' triggers processing"
            )
            return jsonify(
                {
                    "status": "ignored",
                    "task_id": task_id,
                    "event": event_type,
                    "reason": "Only create:task events are processed",
                }
            ), 200
        
        # Process the task in background
        result = get_task_manager().process_async(task_id)
        
        return jsonify(result), 202
    
    except Exception as e:
        logger.error(f"Error handling webhook: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/webhook/process', methods=['POST'])
def manual_process():
    """Manual trigger to process a task."""
    logger.info("Received manual process request")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        task_id = data.get('task_id')
        if not task_id:
            return jsonify({"error": "task_id is required"}), 400
        
        async_mode = data.get('async', True)
        
        if async_mode:
            result = get_task_manager().process_async(task_id)
        else:
            result = get_processor().process_task(task_id)
        
        return jsonify(result), 200 if result['status'] in ['completed', 'processing'] else 202
    
    except Exception as e:
        logger.error(f"Error in manual process: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/status/<int:task_id>', methods=['GET'])
def get_status(task_id):
    """Get processing status for a task."""
    status = get_task_manager().get_status(task_id)
    return jsonify(status)


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CVAT Webhook Receiver for Window Segmentation"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to bind to (default: 5000)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    if not CVAT_USERNAME or not CVAT_PASSWORD:
        logger.error(
            "CVAT_USERNAME and CVAT_PASSWORD environment variables are required!"
        )
        sys.exit(1)
    
    logger.info(f"Starting webhook receiver on {args.host}:{args.port}")
    logger.info(f"CVAT URL: {CVAT_URL}")
    logger.info("Mode: MODEL (automatic window segmentation)")
    logger.info(f"Webhook secret: {'custom' if WEBHOOK_SECRET != 'cvat_webhook_secret' else 'default'}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == "__main__":
    main()
