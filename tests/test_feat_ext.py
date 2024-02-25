import unittest
import cv2
from EVbatteryDetection.vision_system.feature_extraction import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def test_detect_and_compute(self):
        extractor = FeatureExtractor('SIFT')
        img = cv2.imread('data/images/module.jpg', 0)
        keypoints, descriptors = extractor.detect_and_compute(img)
        self.assertTrue(len(keypoints) > 0)
        self.assertIsNotNone(descriptors)

if __name__ == '__main__':
    unittest.main()
