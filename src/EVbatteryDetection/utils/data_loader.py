import json
import numpy as np
import os
from PIL import Image
import supervision as sv


def read_annotations(input_dir):
    annotations = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename)) as f:
                annotations.append(json.load(f))
    return annotations

def polygon_to_mask(polygon, width, height):
    """Convert polygon to binary mask."""
    return sv.polygon_to_mask(polygon.astype(np.int32),(width,height))

def polygon_to_bounding_box(polygon):
    """Calculate bounding box from polygon points."""
    return sv.polygon_to_xyxy(polygon.astype(np.int32))

def process_annotations(input_dir):
    annotations = read_annotations(input_dir)
    im2path = {}
    im2label = {}
    for i, annotation in enumerate(annotations):
        width = annotation['imageWidth']
        height = annotation['imageHeight']
        masks = []
        boxes = []
        im2path[i] = annotation['imagePath']
        im2label[i] = [masks, boxes]
        for shape in annotation['shapes']:
            if shape['shape_type'] == 'polygon':
                # Flatten list of points for the mask function
                polygon = np.array(shape['points'])
                mask = polygon_to_mask(polygon, width, height)
                bounding_box = polygon_to_bounding_box(polygon)
                masks.append(mask)
                boxes.append(bounding_box)
    return im2path, im2label
