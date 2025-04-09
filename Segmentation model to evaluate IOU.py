import json
import os
import cv2
import numpy as np
from pycocotools import mask as maskUtils


def load_coco_annotations(coco_file_path):
    """Load COCO annotations from the given JSON file."""
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def get_ground_truth_mask(coco_annotations, image_id, image_shape):
    """
    Generate a binary ground truth mask from COCO annotations.
    Arguments:
        coco_annotations: COCO dataset object
        image_id: ID of the image to extract annotations for
        image_shape: (height, width) tuple for the image
    Returns:
        binary_mask: Binary mask representing the ground truth segmentation
    """
    # Get annotations for the specific image
    annotations = [
        ann for ann in coco_annotations['annotations'] if ann['image_id'] == image_id
    ]

    # Initialize an empty mask
    binary_mask = np.zeros(image_shape, dtype=np.uint8)

    for ann in annotations:
        # Decode RLE (Run-Length Encoding) or polygon annotations
        if isinstance(ann['segmentation'], list):  # Polygon
            mask = np.zeros(image_shape, dtype=np.uint8)
            for segment in ann['segmentation']:
                pts = np.array(segment).reshape((-1, 2))
                cv2.fillPoly(mask, [np.int32(pts)], color=1)
            binary_mask = np.logical_or(binary_mask, mask).astype(np.uint8)
        elif isinstance(ann['segmentation'], dict):  # RLE
            rle = ann['segmentation']
            mask = maskUtils.decode(rle)
            binary_mask = np.logical_or(binary_mask, mask).astype(np.uint8)
    return binary_mask


def compute_iou(predicted_mask, ground_truth_mask):
    """
    Compute IoU between a predicted mask and a ground truth mask.
    Arguments:
        predicted_mask: Binary predicted mask
        ground_truth_mask: Binary ground truth mask
    Returns:
        iou: Intersection over Union value
    """
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    if union == 0:
        return 0.0  # Avoid division by zero
    return intersection / union


def evaluate_model(predicted_mask_folder, coco_file_path, coco_image_folder):
    """
    Evaluate the segmentation model by computing IoU for each image.
    Arguments:
        predicted_mask_folder: Path to the folder containing predicted binary masks
        coco_file_path: Path to the COCO JSON label file
        coco_image_folder: Path to the folder containing COCO images
    Returns:
        miou: Mean IoU across all images
    """
    coco_annotations = load_coco_annotations(coco_file_path)
    coco_images = coco_annotations['images']

    iou_scores = []

    for image_info in coco_images:
        image_id = image_info['id']
        image_filename = image_info['file_name']

        # Load the corresponding predicted mask
        predicted_mask_path = os.path.join(predicted_mask_folder, image_filename)
        predicted_mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
        if predicted_mask is None:
            print(f"Predicted mask not found for image: {image_filename}")
            continue

        # Ensure binary format for the predicted mask
        predicted_mask = (predicted_mask > 0).astype(np.uint8)

        # Load the ground truth mask
        image_shape = predicted_mask.shape
        ground_truth_mask = get_ground_truth_mask(coco_annotations, image_id, image_shape)

        # Compute IoU
        iou = compute_iou(predicted_mask, ground_truth_mask)
        iou_scores.append(iou)
        print(f"Image: {image_filename}, IoU: {iou:.4f}")

    # Compute mean IoU
    miou = np.mean(iou_scores) if iou_scores else 0.0
    print(f"Mean IoU (mIoU): {miou:.4f}")
    return miou


# Main function to call the evaluation
if __name__ == "__main__":
    # Paths
    predicted_mask_folder = r"Location of your split image file"
    coco_file_path = r"Your coco tag location"
    coco_image_folder = r"File saving path"

    # Evaluate the model
    mean_iou = evaluate_model(predicted_mask_folder, coco_file_path, coco_image_folder)
    print(f"Model Evaluation Complete. mIoU: {mean_iou:.4f}")
