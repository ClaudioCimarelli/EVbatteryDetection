import cv2
import numpy as np
from EVbatteryDetection.vision_system.feature_matching import KorniaMatcher, MatchResult
from EVbatteryDetection.utils.data_loader import load_image, polygon_to_mask, polygon_to_bounding_box, mask_to_polygons
from EVbatteryDetection.utils.data_processing import apply_mask
from EVbatteryDetection.vision_system.image_transformations import HomographyEstimator

from time import perf_counter

class DetectionResult():

    def __init__(self, polygon, template_id, img_shape, match_res: MatchResult= None) -> None:
        self.polygon = polygon
        self.box = polygon_to_bounding_box(polygon)
        self.mask = polygon_to_mask(polygon, *img_shape)
        self.template_id = template_id
        self.match_res = match_res
        self.img_shape = img_shape

class TemplateDetector():

    def __init__(self, templ_imgs, templ_labels, resize_factor=1.) -> None:
        self.templ_labels = templ_labels
        self.templ_imgs = {}
        for i, p in templ_imgs.items():
            mask = self.templ_labels[i][0][0]
            img = load_image(p, color=True)
            if resize_factor < 1:
                # Resize the image
                img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
                # Resize the mask
                mask = cv2.resize(mask.astype(np.uint8), None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
                # Apply threshold to restore boolean consistency
                _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
                mask = mask.astype(bool)
                self.templ_labels[i][0][0] = mask
                self.templ_labels[i][-1] = mask_to_polygons(mask)
                self.templ_labels[i][1] = [polygon_to_bounding_box(p) for p in self.templ_labels[i][-1]]

            self.templ_imgs[i] = apply_mask(img, mask)

        self.img_shape = self.templ_imgs[0].shape[1::-1]

        self.template_score = np.zeros([len(self.templ_imgs)], dtype=np.uint64)

        self.matcher = KorniaMatcher()

    def normalize_scores(self):
        total_score = np.sum(self.template_score)
        if total_score > 0:
            return self.template_score / total_score
        else:
            # If all scores are zero or the total score is zero, return a uniform distribution
            return np.ones(len(self.template_score)) / len(self.template_score)

    def random_select_templ(self):
        normalized_scores = self.normalize_scores()
        permuted_indices = np.random.choice(range(len(self.template_score)), len(self.template_score), replace=False, p=normalized_scores)
        return permuted_indices

    def one_templ_detection(self, path, match_inlrs_thr=150, h_inlrs_thr=.5)-> DetectionResult:

        det_res = None

        img = load_image(path, color=True)
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_AREA)
        image_tensor = self.matcher.imgarray_to_tensor(img, normalize=True, add_batch_dim=True)

        order = self.random_select_templ()
        for t_id in order:
            templ_img = self.templ_imgs[t_id]
            templ_tensor = self.matcher.imgarray_to_tensor(templ_img, normalize=True, add_batch_dim=True)
            # Feature matching
            t1 = perf_counter()
            match_res: MatchResult = self.matcher.match(templ_tensor, image_tensor)
            print(f"Matching took time: {perf_counter()-t1:.2f} secs")
            kps1, kps2, matches = match_res.kps1, match_res.kps2, match_res.matches

            # Homography and perspective transform if enough matches
            if len(matches) > match_inlrs_thr:
                src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                t1 = perf_counter()
                homography_estimator = HomographyEstimator(src_pts, dst_pts)
                print(f"Homography took time: {perf_counter()-t1:.2f} secs")

                h_inlrs = homography_estimator.inliers.astype(bool)[:, 0]
                if h_inlrs.sum()/len(matches) > h_inlrs_thr :
                    # inlr_matches = [m for i, m in enumerate(matches) if h_inlrs[i]]
                    # inlr_kps1 = [kps1[m.queryIdx] for m in inlr_matches]
                    # inlr_kps2 = [kps2[m.trainIdx] for m in inlr_matches]
                    # inlr_match_res = MatchResult(inlr_kps1, inlr_kps2, inlr_matches)

                    templ_poly = self.templ_labels[t_id][-1][0]
                    proj_poly = homography_estimator.transform_points(templ_poly)

                    det_res = DetectionResult(proj_poly, t_id, self.img_shape, match_res)
                    max_inliers = h_inlrs.sum()
                    break


        return det_res
