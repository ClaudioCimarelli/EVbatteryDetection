import cv2

class HomographyEstimator:
    def __init__(self, src_pts, dst_pts, repr_thr=5.0) -> None:
        self.H = None
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=repr_thr)
        self.H = H
        self.mask = mask

    def transform_points(self, points):
        return cv2.perspectiveTransform(points, self.H)
