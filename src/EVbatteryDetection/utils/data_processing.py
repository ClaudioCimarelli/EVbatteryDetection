import cv2
import numpy as np


def apply_mask(image, mask, invert=False):
    """
    Apply a binary mask to an image, hiding everything but the positive bits in the mask.
    Optionally, invert the mask effect to hide the positive bits instead.

    Parameters:
    - image: The input image (numpy array).
    - mask: The binary mask to apply (numpy array of the same size as image).
    - invert: If True, hide the positive bits instead (bool, default False).

    Returns:
    - masked_image: The image after applying the mask.
    """

    masked_image = image.copy()  # Make a copy to avoid altering the original image
    mask = mask.astype(bool)
    if invert:
        # Invert the mask
        masked_image[~mask] = 0  # Set pixels where mask_inv is False to 0

    else:
        # Apply the mask directly
        masked_image[mask] = 0  # Set pixels where mask_inv is False to 0

    return masked_image
