import cv2
import numpy as np
import unittest

# Assuming the process_annotations function is defined elsewhere
from EVbatteryDetection.utils.data_loader import process_annotations, load_image
from EVbatteryDetection.utils.data_processing import apply_mask

class TestApplyMask(unittest.TestCase):
    def test_apply_mask_with_annotations(self):
        # Directory containing images and their annotations
        all_images_dir = "data/images/templates"

        # Process annotations to generate masks (assuming the first item is the target)
        imgs, masks = process_annotations(all_images_dir)
        first_image_mask = masks[0][0][0]  # Adjust indexing based on the actual structure

        # Load the corresponding image (adjust the path as necessary)
        first_image_path = imgs[0]
        image = load_image(first_image_path, color=True)

        # Verify the image and mask are not None
        self.assertIsNotNone(image, "Image could not be loaded.")
        self.assertIsNotNone(first_image_mask, "Mask could not be generated.")

        # Apply the mask to the image
        masked_image = apply_mask(image, first_image_mask, invert=False)

        self.assertEqual(masked_image.shape, image.shape, "Masked image shape mismatch.")

        cv2.imshow('masked img', masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply the mask inversely to the image
        masked_image_inverted = apply_mask(image, first_image_mask, invert=True)

        self.assertEqual(masked_image_inverted.shape, image.shape, "Inverted masked image shape mismatch.")

        cv2.imshow('inverted masked img', masked_image_inverted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
