import cv2
import numpy as np
from EVbatteryDetection.vision_system.feature_matching import KorniaMatcher, MatchResult
from EVbatteryDetection.utils.data_loader import load_image, polygon_to_mask, polygon_to_bounding_box
from EVbatteryDetection.utils.data_processing import apply_mask
from EVbatteryDetection.vision_system.image_transformations import HomographyEstimator


class DetectionResult():

    def __init__(self, polygon, template_img, match_res: MatchResult= None) -> None:
        self.polygon = polygon
        self.box = polygon_to_bounding_box(polygon)
        self.mask = polygon_to_mask(polygon, *template_img.shape[:2][::-1])
        self.templ_img = template_img
        self.match_res = match_res


class TemplateDetector():

    def __init__(self,templ_imgs, templ_labels) -> None:

        self.templ_labels = templ_labels
        self.templ_imgs = {}
        for i, p in templ_imgs.items():
            templ_segm = self.templ_labels[i][0][0]
            self.templ_imgs[i] = apply_mask(load_image(p, color=True), templ_segm)
        self.matcher = KorniaMatcher()


    def one_templ_detection(self, path, match_thr=100)-> DetectionResult:

        det_res = None
        max_inliers = 0

        image_tensor = self.matcher.img_to_tensor(path)

        for t_id, templ_img in self.templ_imgs.items():
            templ_tensor = self.matcher.imgarray_to_tensor(templ_img, normalize=True, add_batch_dim=True)
            # Feature matching
            match_res: MatchResult = self.matcher.match(templ_tensor, image_tensor)

            kps1, kps2, matches = match_res.kps1, match_res.kps2, match_res.matches

            # Homography and perspective transform if enough matches
            if len(matches) > match_thr:
                src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                homography_estimator = HomographyEstimator(src_pts, dst_pts)
                h_inlrs = homography_estimator.inliers.astype(bool)[:, 0]
                if h_inlrs.sum()> max_inliers:
                    # inlr_matches = [m for i, m in enumerate(matches) if h_inlrs[i]]
                    # inlr_kps1 = [kps1[m.queryIdx] for m in inlr_matches]
                    # inlr_kps2 = [kps2[m.trainIdx] for m in inlr_matches]
                    # inlr_match_res = MatchResult(inlr_kps1, inlr_kps2, inlr_matches)

                    templ_poly = self.templ_labels[t_id][-1][0]
                    proj_poly = homography_estimator.transform_points(templ_poly)

                    det_res = DetectionResult(proj_poly, templ_img, match_res)

        return det_res
