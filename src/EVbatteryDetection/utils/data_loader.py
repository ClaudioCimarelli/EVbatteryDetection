import json
import numpy as np
import os
import cv2
import supervision as sv


def read_annotations(input_dir):
    """Reads all annotations JSON from input directory."""
    annotations = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename)) as f:
                annotations.append(json.load(f))
    return annotations

def load_image(image_path, color=False):
    """Load an image from the given path."""
    try:
        if color:
            img = cv2.imread(image_path)
        else:
            img = cv2.imread(image_path, 0)
        return img
    except IOError:
        return None

def polygon_to_mask(polygon, width, height):
    """Convert polygon to binary mask."""
    return sv.polygon_to_mask(polygon.astype(np.int32),(width,height)).astype(bool)

def mask_to_polygons(mask):
    """Convert polygon to binary mask."""
    return sv.mask_to_polygons(mask.astype(bool))

def polygon_to_bounding_box(polygon):
    """Calculate bounding box from polygon points."""
    return sv.polygon_to_xyxy(polygon.astype(np.int32))

def process_annotations(input_dir):
    """Elaborate annotations JSON in path and returns images path dict and labels dict: [masks, boxes, polygons] list for every img id."""
    annotations = read_annotations(input_dir)
    im2path = {}
    im2label = {}
    for i, annotation in enumerate(annotations):
        width = annotation['imageWidth']
        height = annotation['imageHeight']
        masks = []
        boxes = []
        polygons = []
        im2path[i] = os.path.join(input_dir, annotation['imagePath'])
        im2label[i] = [masks, boxes, polygons]
        for shape in annotation['shapes']:
            if shape['shape_type'] == 'polygon':
                # Flatten list of points for the mask function
                polygon = np.array(shape['points'])
                polygons.append(polygon)
                mask = polygon_to_mask(polygon, width, height)
                bounding_box = polygon_to_bounding_box(polygon)
                masks.append(mask)
                boxes.append(bounding_box)
    return im2path, im2label
