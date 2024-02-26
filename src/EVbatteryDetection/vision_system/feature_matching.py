import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import cv2_matches_from_kornia, opencv_kpts_from_laf

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


class KorniaMatcher():

    def __init__(self) -> None:
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        self.lg_matcher = KF.LightGlueMatcher("disk").eval().to(self.device)
        self.num_features = 2048
        self.disk = KF.DISK.from_pretrained("depth").to(self.device)


    def extract_and_match(self, fname1, fname2):
        img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]
        img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]

        hw1 = torch.tensor(img1.shape[2:], device=self.device)
        hw2 = torch.tensor(img2.shape[2:], device=self.device)


        with torch.inference_mode():
            inp = torch.cat([img1, img2], dim=0)
            features1, features2 = self.disk(inp, self.num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors

            lafs1 = KF.laf_from_center_scale_ori(kps1[None])
            lafs2 = KF.laf_from_center_scale_ori(kps2[None])

            dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)

            cv2_matches = cv2_matches_from_kornia(dists, idxs)
            # good_matches = [m for m, n in cv2_matches if m.distance < .7 * n.distance]

        return  opencv_kpts_from_laf(lafs1),  opencv_kpts_from_laf(lafs1), cv2_matches
