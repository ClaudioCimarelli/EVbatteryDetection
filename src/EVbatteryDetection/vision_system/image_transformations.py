import cv2
import numpy as np

class HomographyEstimator:
    def __init__(self, src_pts, dst_pts, repr_thr=5.0) -> None:
        self.H = None
        H, inl = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=repr_thr)
        self.H = H
        self.inliers = inl

    def transform_points(self, points):
        points = np.array(points, np.float32)
        points = points.reshape((-1, 1, 2))
        return cv2.perspectiveTransform(points, self.H).reshape(-1, 2)
