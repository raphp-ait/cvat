import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.segmentation import deeplabv3_resnet50


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


class WindowInferenceCore:
    """Shared two-step inference: segmentation followed by ROI classification."""

    def __init__(
        self,
        segmentation_checkpoint_path: str,
        classifier_checkpoint_path: str,
        device: str | None = None,
        image_size: int = 512,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(image_size)

        self.segmentation_model = self._build_segmentation_model(segmentation_checkpoint_path)
        self.classifier_model, self.class_names = self._build_classifier_model(
            classifier_checkpoint_path
        )

        self.segmentation_preprocess = transforms.Compose(
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

    def _build_segmentation_model(self, checkpoint_path: str):
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _build_classifier_model(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
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

    def infer(self, image: Image.Image, threshold: float = 0.5) -> dict | None:
        rgb_image = image.convert("RGB")
        orig_w, orig_h = rgb_image.size

        input_tensor = self.segmentation_preprocess(rgb_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            seg_output = self.segmentation_model(input_tensor)["out"]
            seg_probs = torch.softmax(seg_output, dim=1)
            window_prob = seg_probs[0, 1].detach().cpu().numpy()

        binary_mask = (window_prob >= float(threshold)).astype(np.uint8)

        # Avoid a hard dependency on cv2 in this shared file.
        resized_binary = np.array(
            Image.fromarray(binary_mask).resize((orig_w, orig_h), resample=Image.NEAREST)
        ).astype(np.uint8)
        resized_prob = np.array(
            Image.fromarray(window_prob).resize((orig_w, orig_h), resample=Image.BILINEAR)
        )

        if int(resized_binary.sum()) < 100:
            return None

        roi = self._crop_and_mask_from_window(np.array(rgb_image), resized_binary)
        cls_input = self.classifier_preprocess(Image.fromarray(roi)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_logits = self.classifier_model(cls_input)
            cls_probs = torch.softmax(cls_logits, dim=1)[0].detach().cpu().numpy()

        pred_idx = int(np.argmax(cls_probs))
        pred_label = self.class_names[pred_idx]
        class_confidence = float(cls_probs[pred_idx])
        region_confidence = float(resized_prob[resized_binary > 0].mean())

        ys, xs = np.where(resized_binary > 0)
        xtl, ytl, xbr, ybr = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        return {
            "binary_mask": resized_binary,
            "bbox": [xtl, ytl, xbr, ybr],
            "label": pred_label,
            "segmentation_confidence": region_confidence,
            "classification_confidence": class_confidence,
            "confidence": region_confidence * class_confidence,
        }
