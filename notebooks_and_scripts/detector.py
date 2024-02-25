import cv2
import numpy as np

# Load images
template_img = cv2.imread('template.jpg', 0)  # Grayscale
test_img = cv2.imread('test.jpg', 0)  # Grayscale

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect and compute SIFT features
keypoints_template, descriptors_template = sift.detectAndCompute(template_img, None)
keypoints_test, descriptors_test = sift.detectAndCompute(test_img, None)

# FLANN parameters and matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_template, descriptors_test, k=2)

# Filter matches using the Lowe's ratio test
good_matches = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

if len(good_matches) > 10:
    # Extract location of good matches
    src_pts = np.float32([ keypoints_template[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_test[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

    # Compute Homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    # Use homography to transform the template corners to test image space
    h, w = template_img.shape
    corners_template = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    corners_test = cv2.perspectiveTransform(corners_template, H)

    # Draw detected polygon on test image
    test_img_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)  # Convert to color to draw in color
    corners_test = np.int32(corners_test)
    cv2.polylines(test_img_color,[corners_test],True,(0,255,0),3, cv2.LINE_AA)

    cv2.imshow('Detected Object', test_img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
