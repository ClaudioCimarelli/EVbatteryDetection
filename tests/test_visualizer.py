import unittest
import cv2
import numpy as np
from EVbatteryDetection.utils.visualizer import *  # Assuming this is your visualizer class
from EVbatteryDetection.vision_system.feature_extraction import OpenCVFeatureExtractor
from EVbatteryDetection.vision_system.feature_matching import OpenCVFeatureMatcher

class TestVisualizer(unittest.TestCase):

    def test_draw_matches(self):
        # Load sample images
        img1 = cv2.imread('data/images/module.jpg', 0)
        img2 = cv2.imread('data/images/templates/video1_00123658.jpg', 0)

        # Initialize FeatureExtractor and FeatureMatcher
        extractor = OpenCVFeatureExtractor(method='SIFT')
        matcher = OpenCVFeatureMatcher(method='BF')

        # Extract features
        kp1, des1 = extractor.detect_and_compute(img1)
        kp2, des2 = extractor.detect_and_compute(img2)

        # Match features
        matches = matcher.match(des1, des2)

        # Test draw_matches method
        try:
            draw_matches(img1, kp1, img2, kp2, matches, True)
            print("Visualization successful. Please inspect the output visually.")
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()
