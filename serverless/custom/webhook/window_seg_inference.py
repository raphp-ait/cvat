import os
import importlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_window_inference_core():
    try:
        module = importlib.import_module("window_inference_shared")
        return module.WindowInferenceCore
    except ModuleNotFoundError:
        pass

    shared_file = (
        Path(__file__).resolve().parents[2]
        / "pytorch"
        / "custom"
        / "window_seg"
        / "nuclio"
        / "window_inference_shared.py"
    )
    spec = importlib.util.spec_from_file_location("window_inference_shared", shared_file)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load shared inference module from {shared_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WindowInferenceCore


WindowInferenceCore = _load_window_inference_core()


DEFAULT_THRESHOLD = 0.5


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
        self.core = WindowInferenceCore(
            segmentation_checkpoint_path=str(self.checkpoint_path),
            classifier_checkpoint_path=str(self.classifier_checkpoint_path),
            device=device,
            image_size=self.image_size,
        )
        self.device = self.core.device

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
        inference = self.core.infer(image, threshold=active_threshold)
        if inference is None:
            return []

        cvat_mask = self._to_cvat_rle(inference["binary_mask"])
        if not cvat_mask:
            return []
        return [
            {
                "confidence": str(inference["confidence"]),
                "segmentation_confidence": str(inference["segmentation_confidence"]),
                "classification_confidence": str(inference["classification_confidence"]),
                "label": inference["label"],
                "mask": cvat_mask,
                "type": "mask",
            }
        ]
