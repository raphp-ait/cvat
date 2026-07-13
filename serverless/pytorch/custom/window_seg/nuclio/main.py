import json
import base64
import io

import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image


CHECKPOINT_PATH = "/opt/nuclio/window_seg_best.pth"
CLASSIFIER_CHECKPOINT_PATH = "/opt/nuclio/window_cls_best.pth"
CONFIDENCE_THRESHOLD = 0.5
DEFAULT_WINDOW_CLASSES = [
    "Aluminium",
    "Bauschutt",
    "Beton",
    "Erde",
    "Feuerfest Zement",
    "Filterkerzenbruch",
    "Glas",
    "Kalk",
    "Keramik",
    "Metall",
    "Metall Eisen",
    "Mineralwolle",
    "Schamott",
    "Schlamm",
    "Schotter/Sand",
]


def to_cvat_mask(box, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened


def crop_and_mask_from_window(image_np, window_mask):
    ys, xs = np.where(window_mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return image_np

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    crop = image_np[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = window_mask[y1 : y2 + 1, x1 : x2 + 1]
    crop[crop_mask == 0] = 0
    return crop


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

    cls_checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if isinstance(cls_checkpoint, dict) and "state_dict" in cls_checkpoint:
        cls_state_dict = cls_checkpoint["state_dict"]
        class_names = cls_checkpoint.get("classes") or DEFAULT_WINDOW_CLASSES
    else:
        cls_state_dict = cls_checkpoint
        class_names = DEFAULT_WINDOW_CLASSES

    cls_model = resnet18(weights=None)
    cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, len(class_names))
    cls_model.load_state_dict(cls_state_dict)
    cls_model.to(device)
    cls_model.eval()

    context.user_data.classifier = cls_model
    context.user_data.class_names = [str(name) for name in class_names]
    context.user_data.device = device
    context.user_data.img_size = 512
    context.user_data.transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    context.user_data.classifier_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
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
    classifier = context.user_data.classifier
    class_names = context.user_data.class_names
    device = context.user_data.device
    transform = context.user_data.transform
    classifier_preprocess = context.user_data.classifier_preprocess

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

    roi = crop_and_mask_from_window(np.array(image), binary_mask)
    roi_pil = Image.fromarray(roi)
    cls_input = classifier_preprocess(roi_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_logits = classifier(cls_input)
        cls_probs = torch.softmax(cls_logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(cls_probs))
    pred_label = class_names[pred_idx]
    class_confidence = float(cls_probs[pred_idx])

    region_confidence = float(window_prob_resized[binary_mask > 0].mean())
    results.append({
        "confidence": str(region_confidence * class_confidence),
        "segmentation_confidence": str(region_confidence),
        "classification_confidence": str(class_confidence),
        "label": pred_label,
        "mask": cvat_mask,
        "type": "mask",
    })

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
