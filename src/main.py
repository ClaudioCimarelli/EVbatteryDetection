from EVbatteryDetection.vision_system.feature_extraction import OpenCVFeatureExtractor
from EVbatteryDetection.vision_system.feature_matching import OpenCVFeatureMatcher
from EVbatteryDetection.vision_system.image_transformations import HomographyEstimator
from EVbatteryDetection.utils.visualizer import *
from EVbatteryDetection.utils.data_loader import process_annotations, load_image, polygon_to_mask
from EVbatteryDetection.utils.data_processing import apply_mask
import cv2
import numpy as np

def main():
    # Initialize modules
    extractor = OpenCVFeatureExtractor(method='SIFT')
    matcher = OpenCVFeatureMatcher(method='FLANN')

    # Load templates
    templates_dir = 'data/images/templates'
    templ_imgs, templ_labels = process_annotations(templates_dir)
    all_images_dir = 'data/images/all'
    all_imgs, all_labels = process_annotations(all_images_dir)

    template_img = load_image(templ_imgs[0])
    image_size = template_img.shape[::-1]
    templ_segm = templ_labels[0][0][0]
    masked_templ = apply_mask(template_img, templ_segm)
    templ_poly = templ_labels[0][-1][0]
    test_img = load_image(all_imgs[0])  # Grayscale

    # Feature extraction
    keypoints_template, descriptors_template = extractor.detect_and_compute(masked_templ)
    keypoints_test, descriptors_test = extractor.detect_and_compute(test_img)

    # Feature matching
    good_matches = matcher.match(descriptors_template, descriptors_test, ratio_thresh=.8)

    draw_matches(masked_templ, keypoints_template, test_img, keypoints_test, good_matches, True)

    # Homography and perspective transform if enough matches
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography_estimator = HomographyEstimator(src_pts, dst_pts)
        h_inlrs = homography_estimator.inliers.astype(bool)
        draw_matches(masked_templ, keypoints_template, test_img, keypoints_test, [m for i, m in enumerate(good_matches) if h_inlrs[i]], True)
        pts_inlrs = dst_pts[homography_estimator.inliers.astype(bool)]
        projected_templ = homography_estimator.transform_points(templ_poly)
        # Visualization
        test_img_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)  # For visualization
        out = overlay_mask_with_border(test_img_color, polygon_to_mask(projected_templ, *image_size), polygon=projected_templ, draw_now=True)

    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))

if __name__ == "__main__":
    main()
