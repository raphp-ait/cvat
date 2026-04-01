# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam2_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_base_plus.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        sam2_model = build_sam2(
            self.model_cfg,
            self.sam2_checkpoint,
            device=self.device,
        )
        self.predictor = SAM2ImagePredictor(sam2_model)

    def handle(self, image, pos_points, neg_points, obj_bbox=None, threshold=0.5):
        image_np = np.array(image)
        self.predictor.set_image(image_np)

        # Build point_coords and point_labels arrays
        # pos_points: list of [x, y], label=1 (foreground)
        # neg_points: list of [x, y], label=0 (background)
        all_points = []
        all_labels = []
        for pt in pos_points:
            all_points.append([pt[0], pt[1]])
            all_labels.append(1)
        for pt in neg_points:
            all_points.append([pt[0], pt[1]])
            all_labels.append(0)

        point_coords = np.array(all_points, dtype=np.float32) if all_points else None
        point_labels = np.array(all_labels, dtype=np.int32) if all_labels else None

        # Build box prompt if provided
        # obj_bbox comes as [[x1, y1], [x2, y2]] from CVAT
        box = None
        if obj_bbox and len(obj_bbox) >= 2:
            x1, y1 = obj_bbox[0]
            x2, y2 = obj_bbox[1]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Use multimask_output when few points (ambiguous prompt),
        # single mask when many points (user is refining)
        use_multimask = (len(all_points) <= 2)

        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=use_multimask,
            normalize_coords=True,
        )

        # Select best mask by highest predicted IoU
        best_idx = int(np.argmax(iou_predictions))
        best_mask = masks[best_idx]  # shape: (H, W), dtype: bool

        # Convert to uint8 0/255 mask
        mask_uint8 = (best_mask.astype(np.uint8)) * 255

        # Compute tight bounding box of the mask for the 'bounds' field
        ys, xs = np.where(mask_uint8 > 0)
        bounds = None
        if len(xs) > 0 and len(ys) > 0:
            left, top = int(xs.min()), int(ys.min())
            right, bottom = int(xs.max()), int(ys.max())
            bounds = [left, top, right, bottom]
            # Crop mask to bounding box to reduce payload size
            mask_uint8 = mask_uint8[top:bottom + 1, left:right + 1]

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return mask_uint8, bounds