import json
import base64
import io

import numpy as np
from PIL import Image
from window_inference_shared import WindowInferenceCore


CHECKPOINT_PATH = "/opt/nuclio/window_seg_best.pth"
CLASSIFIER_CHECKPOINT_PATH = "/opt/nuclio/window_cls_best.pth"
CONFIDENCE_THRESHOLD = 0.5


def to_cvat_mask(box, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened


def init_context(context):
    context.logger.info("Init context...  0%")

    engine = WindowInferenceCore(
        segmentation_checkpoint_path=CHECKPOINT_PATH,
        classifier_checkpoint_path=CLASSIFIER_CHECKPOINT_PATH,
    )
    context.logger.info(f"Using device: {engine.device}")
    context.user_data.engine = engine

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run window_seg model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", CONFIDENCE_THRESHOLD))
    image = Image.open(buf).convert("RGB")

    results = []
    inference = context.user_data.engine.infer(image, threshold=threshold)
    if inference is None:
        # No significant window region detected
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    mask_full = (inference["binary_mask"] * 255).astype(np.uint8)
    cvat_mask = to_cvat_mask(inference["bbox"], mask_full)
    results.append({
        "confidence": str(inference["confidence"]),
        "segmentation_confidence": str(inference["segmentation_confidence"]),
        "classification_confidence": str(inference["classification_confidence"]),
        "label": inference["label"],
        "mask": cvat_mask,
        "type": "mask",
    })

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
