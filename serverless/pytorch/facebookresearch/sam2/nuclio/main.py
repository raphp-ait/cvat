# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import json
import base64
from PIL import Image
import io
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    pos_points = data.get("pos_points", [])
    neg_points = data.get("neg_points", [])
    obj_bbox = data.get("obj_bbox", None)
    threshold = data.get("threshold", 0.5)

    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")

    mask, bounds = context.user_data.model.handle(
        image, pos_points, neg_points, obj_bbox, threshold
    )

    results = {"mask": mask.tolist()}
    if bounds is not None:
        results["bounds"] = bounds

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type='application/json',
        status_code=200
    )