import unittest
import numpy as np
from EVbatteryDetection.vision_system.image_transformations import HomographyEstimator

class TestHomographyComputer(unittest.TestCase):
    def test_compute_homography(self):
        src_pts = np.float32([[10, 10], [100, 10], [100, 100], [10, 100]]).reshape(-1, 1, 2)
        dst_pts = src_pts + 10

        h_est = HomographyEstimator(src_pts, dst_pts)
        self.assertIsNotNone(h_est.H)

if __name__ == '__main__':
    unittest.main()
