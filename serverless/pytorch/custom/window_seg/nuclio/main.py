import json
import base64
import io

import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from skimage.measure import find_contours, approximate_polygon


CHECKPOINT_PATH = "/opt/nuclio/window_seg_best.pth"
CONFIDENCE_THRESHOLD = 0.5


def to_cvat_mask(box, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened


def init_context(context):
    context.logger.info("Init context...  0%")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    context.logger.info(f"Using device: {device}")

    # Load DeepLabV3-ResNet50 — must match training construction:
    # pretrained backbone with classifier/aux_classifier heads replaced for 2 classes
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    context.user_data.model = model
    context.user_data.device = device
    context.user_data.img_size = 512
    context.user_data.transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run window_seg model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", CONFIDENCE_THRESHOLD))
    image = Image.open(buf).convert("RGB")
    orig_w, orig_h = image.size

    model = context.user_data.model
    device = context.user_data.device
    transform = context.user_data.transform

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)["out"]
        probs = torch.softmax(output, dim=1)
        # Class 1 = window
        window_prob = probs[0, 1].cpu().numpy()

    # Threshold to binary mask
    binary_mask = (window_prob >= threshold).astype(np.uint8)

    # Resize prediction back to original image size
    binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    window_prob_resized = cv2.resize(window_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    results = []
    if binary_mask.sum() < 100:
        # No significant window region detected
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    # Return single mask for the entire window region
    ys, xs = np.where(binary_mask > 0)
    xtl, ytl, xbr, ybr = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    mask_full = (binary_mask * 255).astype(np.uint8)
    cvat_mask = to_cvat_mask([xtl, ytl, xbr, ybr], mask_full)

    contours = find_contours(binary_mask, 0.5)
    if contours:
        contour = max(contours, key=len)
        contour = np.flip(contour, axis=1)  # (row,col) -> (x,y)
        contour = approximate_polygon(contour, tolerance=2.5)
        if len(contour) >= 3:
            region_confidence = float(window_prob_resized[binary_mask > 0].mean())
            results.append({
                "confidence": str(region_confidence),
                "label": "Leerraum",
                "points": contour.ravel().tolist(),
                "mask": cvat_mask,
                "type": "mask",
            })

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
