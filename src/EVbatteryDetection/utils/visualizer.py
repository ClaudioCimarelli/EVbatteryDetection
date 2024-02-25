import cv2
import numpy as np

class Visualizer:
    def draw_matches(self, img1, keypoints1, img2, keypoints2, matches):
        out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
        cv2.imshow('Detected Object', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_detected_object(self, img, corners):
        cv2.polylines(img, [np.int32(corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Detected Object', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
