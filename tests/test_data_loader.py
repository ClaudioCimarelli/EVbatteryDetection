import unittest
import numpy as np
from EVbatteryDetection.utils.data_loader import *

class TestDataLoader(unittest.TestCase):

    def test_read_annotations(self):
        path = 'data/images/train'
        annot = read_annotations(path)
        self.assertTrue(len(annot) == 25)
        path = 'data/images/test'
        annot = read_annotations(path)
        self.assertTrue(len(annot) == 10)

    def test_polygon_to_mask(self):
        image_shape = (100, 100)
        polygon = np.array([(0,0), (0, 25), (25, 25), (25, 0)])
        mask = polygon_to_mask(polygon, *image_shape)
        self.assertTrue(mask.sum() == 26*26, "Mask should contain filled area.")

    def test_polygons_to_bounding_boxes(self):
        polygon = np.array([(25, 25), (75, 25), (75, 75), (25, 75)])
        bounding_box = polygon_to_bounding_box(polygon)
        expected_box = np.array([25, 25, 75, 75])
        np.testing.assert_array_equal(bounding_box, expected_box, "Bounding box coordinates are incorrect.")

    def test_process_annotations(self):
        path = 'data/images/train'
        paths, labels = process_annotations(path)
        self.assertTrue(len(paths) == 25)
        self.assertTrue(len(labels) == 25)
        path = 'data/images/test'
        paths, labels = process_annotations(path)
        self.assertTrue(len(paths) == 10)
        self.assertTrue(len(labels) == 10)

if __name__ == '__main__':
    unittest.main()
