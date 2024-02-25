import cv2

class FeatureExtractor:
    def __init__(self, method='SIFT'):
        if method == 'SIFT':
            self.detector = cv2.SIFT_create()
        else:
            self.detector = cv2.ORB_create()

    def detect_and_compute(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
