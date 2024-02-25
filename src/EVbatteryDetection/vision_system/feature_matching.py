import cv2
import numpy as np
import kornia

class OpenCVFeatureMatcher:
    def __init__(self, method='FLANN'):
        if method == 'FLANN':
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif method == 'BF':
            self.matcher = cv2.BFMatcher()

    def match(self, descriptors1, descriptors2, ratio_thresh=0.8):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        return good_matches
