import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.segmentation import deeplabv3_resnet50


DEFAULT_THRESHOLD = 0.5
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


class WindowSegInference:
    """Reusable inference wrapper for the window segmentation model."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        classifier_checkpoint_path: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        image_size: int = 512,
        device: str | None = None,
        label_name: str = "Leerraum",
    ) -> None:
        self.threshold = float(threshold)
        self.image_size = int(image_size)
        self.label_name = label_name

        if checkpoint_path is None:
            checkpoint_path = os.getenv("WINDOW_SEG_CHECKPOINT", "/app/window_seg_best.pth")

        if classifier_checkpoint_path is None:
            classifier_checkpoint_path = os.getenv("WINDOW_CLS_CHECKPOINT", "/app/window_cls_best.pth")

        self.checkpoint_path = self._resolve_checkpoint_path(Path(checkpoint_path))
        self.classifier_checkpoint_path = self._resolve_checkpoint_path(
            Path(classifier_checkpoint_path),
            candidate_names=["window_cls_best.pth"],
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model()
        self.classifier, self.class_names = self._build_classifier()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )
        self.classifier_preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _resolve_checkpoint_path(
        self,
        checkpoint_path: Path,
        candidate_names: list[str] | None = None,
    ) -> Path:
        def resolve_file_or_dir(path: Path) -> Path | None:
            if path.is_file():
                return path

            if path.is_dir():
                pth_files = sorted(path.glob("*.pth"))
                if pth_files:
                    return pth_files[0]

            return None

        module_dir = Path(__file__).resolve().parent
        candidates = [checkpoint_path]
        candidate_names = candidate_names or ["window_seg_best.pth"]

        if not checkpoint_path.is_absolute():
            candidates.extend(
                [
                    Path.cwd() / checkpoint_path,
                    module_dir / checkpoint_path,
                ]
            )

        # Common local paths for this repository layout.
        for candidate_name in candidate_names:
            candidates.extend(
                [
                    module_dir / candidate_name,
                    module_dir / candidate_name / candidate_name,
                    module_dir.parent.parent / f"pytorch/custom/window_seg/nuclio/{candidate_name}",
                    Path.cwd() / f"cvat/serverless/pytorch/custom/window_seg/nuclio/{candidate_name}",
                    Path.cwd() / f"cvat/serverless/custom/webhook/{candidate_name}",
                    Path.cwd() / f"cvat/serverless/custom/webhook/{candidate_name}/{candidate_name}",
                ]
            )

        # Preserve order while deduplicating.
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str in seen:
                continue
            seen.add(candidate_str)
            unique_candidates.append(candidate)

        for candidate in unique_candidates:
            resolved = resolve_file_or_dir(candidate)
            if resolved is not None:
                return resolved

        tried = "\n - " + "\n - ".join(str(p) for p in unique_candidates)
        raise FileNotFoundError(
            "Checkpoint not found. Tried these paths:" + tried
        )

    def _build_model(self):
        # Use only the local checkpoint; avoid runtime downloads of pretrained weights.
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

        state_dict = torch.load(
            str(self.checkpoint_path),
            map_location=self.device,
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _build_classifier(self):
        checkpoint = torch.load(
            str(self.classifier_checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            class_names = checkpoint.get("classes") or DEFAULT_WINDOW_CLASSES
        else:
            state_dict = checkpoint
            class_names = DEFAULT_WINDOW_CLASSES

        class_names = [str(name) for name in class_names]
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model, class_names

    @staticmethod
    def _crop_and_mask_from_window(image_np: np.ndarray, window_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(window_mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return image_np

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        crop = image_np[y1 : y2 + 1, x1 : x2 + 1].copy()
        crop_mask = window_mask[y1 : y2 + 1, x1 : x2 + 1]
        crop[crop_mask == 0] = 0
        return crop

    @staticmethod
    def _to_cvat_rle(mask: np.ndarray) -> list[float]:
        """
        Convert a binary mask to CVAT RLE format.

        CVAT expects run-length values followed by [left, top, right, bottom].
        """
        inds = np.argwhere(mask > 0)
        if len(inds) <= 1:
            return []

        top, left = np.min(inds, axis=0)
        bottom, right = np.max(inds, axis=0)
        top, left, bottom, right = int(top), int(left), int(bottom), int(right)

        local_mask = mask[top : bottom + 1, left : right + 1].astype(np.uint8).flatten()

        rle = []
        prev = 0
        count = 0
        for value in local_mask:
            if prev != value:
                rle.append(count)
                count = 0
            count += 1
            prev = int(value)

        rle.extend([count, left, top, right, bottom])
        return [float(v) for v in rle]

    def segment_image(self, image: Image.Image, threshold: float | None = None) -> list[dict]:
        """
        Segment a PIL image and return mask-only CVAT payload entries.

        Returns list of dicts with keys: confidence, label, mask, type.
        """
        active_threshold = self.threshold if threshold is None else float(threshold)
        rgb_image = image.convert("RGB")
        orig_w, orig_h = rgb_image.size

        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"]
            probs = torch.softmax(output, dim=1)
            window_prob = probs[0, 1].detach().cpu().numpy()

        binary_mask = (window_prob >= active_threshold).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        window_prob_resized = cv2.resize(window_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        if int(binary_mask.sum()) < 100:
            return []

        cvat_mask = self._to_cvat_rle(binary_mask)
        if not cvat_mask:
            return []

        roi = self._crop_and_mask_from_window(np.array(rgb_image), binary_mask)
        roi_pil = Image.fromarray(roi)
        cls_input = self.classifier_preprocess(roi_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_logits = self.classifier(cls_input)
            cls_probs = torch.softmax(cls_logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(cls_probs))
        pred_label = self.class_names[pred_idx]
        class_confidence = float(cls_probs[pred_idx])

        region_confidence = float(window_prob_resized[binary_mask > 0].mean())
        return [
            {
                "confidence": str(region_confidence * class_confidence),
                "segmentation_confidence": str(region_confidence),
                "classification_confidence": str(class_confidence),
                "label": pred_label,
                "mask": cvat_mask,
                "type": "mask",
            }
        ]
