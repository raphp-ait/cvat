#!/usr/bin/env python3
"""
Test script for CVAT SDK integration
Run this inside the webhook container to test SDK functionality
Uses the low-level API as per official CVAT SDK documentation
"""

import os
import sys

from PIL import Image

from window_seg_inference import WindowSegInference

# CVAT SDK - Low-level API
from cvat_sdk.api_client import ApiClient, Configuration
from cvat_sdk.api_client.models import (
    LabeledShapeRequest,
    PatchedLabeledDataRequest,
    ShapeType,
)
from cvat_sdk.api_client.api import jobs_api

from urllib.parse import urlparse, parse_qs

# Configuration
CVAT_URL = os.getenv("CVAT_URL", "http://ls1hvs03:8080")
CVAT_USERNAME = os.getenv("CVAT_USERNAME", "admin")
CVAT_PASSWORD = os.getenv("CVAT_PASSWORD", "Changeme!")

PROJECT_NAME = "IZ_Annotation"
PAGE_SIZE = 999

SEGMENTATION_THRESHOLD = float(os.getenv("SEGMENTATION_THRESHOLD", "0.5"))

print(f"CVAT URL: {CVAT_URL}")
print(f"Username: {CVAT_USERNAME}")
print(f"Password: {'***' if CVAT_PASSWORD else 'NOT SET'}")
print("=" * 60)

_WINDOW_SEG_MODEL = None

def test_connection():
    """Test basic connection to CVAT"""
    print("\n[1/6] Testing connection...")
    try:
        configuration = Configuration(
            host=CVAT_URL,
            username=CVAT_USERNAME,
            password=CVAT_PASSWORD,
        )
        api_client = ApiClient(configuration)
        print("✓ ApiClient created successfully")
        return api_client
    except Exception as e:
        print(f"✗ Failed to create ApiClient: {e}")
        return None

def get_from_paginated_list(api, **kwargs) -> list:
    """
    Retrieves all items from a paginated list using the provided API.

    Args:
        api: The API object used to make API requests.
        **kwargs: Additional keyword arguments passed to the API's list method.

    Returns:
        list: All items retrieved from the paginated list.
    """
    results = []
    data, _ = api.list(page_size=100, **kwargs)
    results += data['results']
    while data['next']:
        next_page = parse_qs(urlparse(data['next']).query)
        next_page = int(next_page['page'][0])
        data, _ = api.list(page_size=100, page=next_page, **kwargs)
        results += data['results']
    
    return results


def get_project(api_client: ApiClient,
                project_name: str):
    """
    Get project from CVAT and check it.

    Args:
        api_client (ApiClient): CVAT API client.
        project_name (str): Project name in CVAT.

    Raises:
        ValueError: Project does not exists.
        ValueError: Project is not unique.

    Returns:
        dict: CVAT project.
    """

    # CVAT does only substring matching
    projects = get_from_paginated_list(api_client.projects_api, search=project_name)
    if len(projects) == 0:
        raise ValueError(f'Project {project_name} does not exist.')
    
    exact_matched_projects = []
    for project in projects:
        if project.name == project_name:
            exact_matched_projects.append(project)

    if len(exact_matched_projects) > 1:
        raise ValueError(f'Project {project_name} is not unique.')
    
    return exact_matched_projects[0]


def get_labels(api_client: ApiClient,
               project_id: int) -> list:
    """
    Get labels from given CVAT project ID.

    Args:
        api_client (ApiClient): CVAT API client.
        project_id (int): Project ID in CVAT.

    Returns:
        list: List of labels.
    """
    project_labels = get_from_paginated_list(api=api_client.labels_api, project_id=project_id)
    labels = []
    for label in project_labels:
        label = vars(label)['_data_store']
        labels.append(label.copy())
        
        labels[-1]['attributes'] = []
        for attr in label['attributes']:
            attr = vars(attr)['_data_store']
            attr['input_type'] = str(attr['input_type'])
            labels[-1]['attributes'].append(attr)
    
    return labels



def get_image(
    api_client: ApiClient,
    job_id: int,
    frame_number: int,
) -> Image.Image:
    """
    Retrieve an individual frame from a CVAT job.
    """
    jobs_api_instance = jobs_api.JobsApi(api_client)

    image_data, _ = jobs_api_instance.retrieve_data(
        job_id,
        type="frame",
        number=frame_number,
        quality="original",
    )

    image_data.seek(0)

    with Image.open(image_data) as image:
        return image.copy()


def process_image(image: Image.Image, frame_number: int, label_id: int) -> list[dict]:
    """
    Run window segmentation model and return CVAT-style shape annotations.

    Uses the same window_seg_best.pth model setup as the Nuclio function.
    """
    global _WINDOW_SEG_MODEL

    if _WINDOW_SEG_MODEL is None:
        checkpoint_path = os.getenv("WINDOW_SEG_CHECKPOINT", "/app/window_seg_best.pth")
        _WINDOW_SEG_MODEL = WindowSegInference(
            checkpoint_path=checkpoint_path,
            threshold=SEGMENTATION_THRESHOLD,
        )

    model_results = _WINDOW_SEG_MODEL.segment_image(image)

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


def update_annotations(
    api_client: ApiClient,
    job_id: int,
    frame_number: int,
    annotations: list[dict],
) -> None:
    """
    Append annotations for a single frame in a job.

    This keeps existing annotations and only creates new shapes.
    """
    jobs_api_instance = jobs_api.JobsApi(api_client)

    if not annotations:
        return

    # Create new shapes for this frame.
    shapes_to_create = []
    for ann in annotations:
        shape_type = ann.get("type", "rectangle")
        if isinstance(shape_type, str):
            shape_type = ShapeType(shape_type)

        shape_request = LabeledShapeRequest(
            type=shape_type,
            frame=ann["frame"],
            label_id=ann["label_id"],
            points=ann["points"],
            group=ann.get("group", 0),
            source=ann.get("source", "auto"),
            occluded=ann.get("occluded", False),
            outside=ann.get("outside", False),
            z_order=ann.get("z_order", 0),
            rotation=ann.get("rotation", 0),
            attributes=ann.get("attributes", []),
        )
        shapes_to_create.append(shape_request)

    create_request = PatchedLabeledDataRequest(shapes=shapes_to_create)
    jobs_api_instance.partial_update_annotations(
        id=job_id,
        action="create",
        patched_labeled_data_request=create_request,
    )


def main():
    """Run all tests"""
    if not CVAT_USERNAME or not CVAT_PASSWORD:
        print("ERROR: CVAT_USERNAME and CVAT_PASSWORD environment variables are required!")
        sys.exit(1)
    
    # Test 1: Connection (includes authentication)
    api_client = test_connection()
    if not api_client:
        sys.exit(1)
    
    try:
        # Use default task ID or from environment
        task_id = int(os.getenv("TEST_TASK_ID", "577"))
        print(f"\nUsing task ID: {task_id}")
        
        
        # Test 3: Get labels
        labels = get_labels(api_client, get_project(api_client, PROJECT_NAME)["id"])
        if not labels:
            print("No labels found in project; cannot upload annotations.")
            return

        default_label_id = labels[0]["id"]
        label_id = int(os.getenv("TEST_LABEL_ID", str(default_label_id)))
        print(f"Using label ID: {label_id}")
        
        
        jobs, _ =  api_client.jobs_api.list(search=PROJECT_NAME, page_size=PAGE_SIZE)
        # Test 5: Get frames (if jobs exist)
        for job in jobs["results"]:
            if job["task_id"] != task_id:
                continue

            job_id = job["id"]

            #get frames
            data_meta, _ = api_client.jobs_api.retrieve_data_meta(job_id)
            frames = data_meta["frames"]

            for frame_number, _ in enumerate(frames):
                pil_image = get_image(api_client, job_id, frame_number)
                annotations = process_image(pil_image, frame_number, label_id)
                update_annotations(api_client, job_id, frame_number, annotations)
                print(f"Updated job {job_id}, frame {frame_number} with {len(annotations)} annotation(s)")
            

    
    finally:
        # Cleanup
        api_client.close()

if __name__ == "__main__":
    main()
