import unittest
import cv2
from EVbatteryDetection.vision_system.feature_extraction import OpenCVFeatureExtractor
from EVbatteryDetection.vision_system.feature_matching import OpenCVFeatureMatcher, KorniaMatcher
from EVbatteryDetection.utils.visualizer import draw_matches

class TestFeatureMatcher(unittest.TestCase):

    def test_flann_matcher(self):
        extractor = OpenCVFeatureExtractor('SIFT')
        matcher = OpenCVFeatureMatcher('FLANN')

        img1 = cv2.imread('data/images/module.jpg', 0)
        img2 = cv2.imread('data/images/templates/video1_00123658.jpg', 0)

        kp1, des1 = extractor.detect_and_compute(img1)
        kp2, des2 = extractor.detect_and_compute(img2)

        matches = matcher.match(des1, des2)
        self.assertTrue(len(matches) > 0)

    def test_bf_matcher(self):
        extractor = OpenCVFeatureExtractor('SIFT')
        matcher = OpenCVFeatureMatcher('BF')

        img1 = cv2.imread('data/images/module.jpg', 0)
        img2 = cv2.imread('data/images/templates/video1_00123658.jpg', 0)

        kp1, des1 = extractor.detect_and_compute(img1)
        kp2, des2 = extractor.detect_and_compute(img2)

        matches = matcher.match(des1, des2)
        self.assertTrue(len(matches) > 0)

    def test_kornia_matcher(self):
        matcher = KorniaMatcher()

        img1 ='data/images/all/video1_00111121.jpg'
        img2 = 'data/images/all/video1_00111219.jpg'
        kps1, kps2, matches = matcher.extract_and_match(img1, img2)
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        draw_matches(img1, kps1, img2, kps2, matches, True)
        self.assertTrue(len(matches) > 0)

if __name__ == '__main__':
    unittest.main()
