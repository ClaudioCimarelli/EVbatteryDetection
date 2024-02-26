import cv2
import numpy as np

def draw_matches(img1, keypoints1, img2, keypoints2, matches, draw_now=False):
    out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if draw_now:
        cv2.imshow('Matched Keypoints', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out

def draw_detected_object(img, polygon, border_color=(0, 0, 255), border_thickness=3, draw_now=False):
    """
    Draw a polygon on an image as a representation of a detected object's border.

    Parameters:
    - img: Image on which to draw the polygon (numpy array).
    - polygon: List of (x, y) tuples representing the polygon's vertices.
    - border_color: Tuple representing the color of the polygon border (BGR format).
    - border_thickness: Thickness of the polygon border.
    - draw_now: If True, display the image immediately after drawing.

    The function modifies the input image in place.
    """
    # Convert the polygon points to a format suitable for cv2.polylines
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Draw the polygon on the image
    out = cv2.polylines(img, [polygon], isClosed=True, color=border_color, thickness=border_thickness, lineType=cv2.LINE_AA)

    if draw_now:
        cv2.imshow('Detected Object', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out

def overlay_mask_with_border(image, mask, polygon=None, color=(0, 255, 0), border_color=(0, 0, 255), alpha=0.5, border_thickness=2, draw_now=False):
    """
    Overlay a segmentation mask on a color image with a specified transparent color and add a solid color border.

    Parameters:
    - image: Color image (numpy array) on which to overlay the mask.
    - mask: Binary mask (numpy array) indicating the segmented area.
    - polygon: List of (x, y) tuples representing the points of the polygon from which the mask was generated (optional).
    - color: Tuple representing the BGR color of the overlay (default is green).
    - border_color: Tuple representing the BGR color of the border (default is blue).
    - alpha: Float representing the transparency of the overlay color (default is 0.5).
    - border_thickness: Integer representing the thickness of the border line (default is 2).

    Returns:
    - overlayed_image: Color image with the mask overlayed and border added.
    """
    # Create an overlay image with the specified color where the mask is positive
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):  # Assuming BGR format
        overlay[:, :, i] = mask * color[i]

    # Blend the overlay with the original image
    overlayed_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw the polygon (border) on the overlayed
    if polygon is not None:
        overlayed_image = draw_detected_object(overlayed_image, polygon, border_color=border_color, border_thickness=border_thickness)

    if draw_now:
        cv2.imshow('Detected Object Mask', overlayed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return overlayed_image
