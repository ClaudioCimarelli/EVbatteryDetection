import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import cv2_matches_from_kornia, opencv_kpts_from_laf


class MatchResult:

    def __init__(self, kps1, kps2, matches) -> None:
        self.kps1 = kps1
        self.kps2 = kps2
        self.matches = matches

class OpenCVFeatureMatcher:
    def __init__(self, method='FLANN'):
        if method == 'FLANN':
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif method == 'BF':
            self.matcher = cv2.BFMatcher()

    def match(self, kps1, kps2, descriptors1, descriptors2, ratio_thresh=0.8):
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        match_res = MatchResult(kps1, kps2, good_matches)
        return match_res


class KorniaMatcher():

    def __init__(self) -> None:
        self.device = K.utils.get_cuda_or_mps_device_if_available()
        self.lg_matcher = KF.LightGlueMatcher("disk").eval().to(self.device)
        self.num_features = 2048
        self.disk = KF.DISK.from_pretrained("depth").to(self.device)

    def img_to_tensor(self, fname1)->torch.Tensor:
        img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]
        return img1

    def imgarray_to_tensor(self, array: np.array, normalize=False, add_batch_dim=False)->torch.Tensor:
        tensor = K.utils.image_to_tensor(array, keepdim=not add_batch_dim).to(self.device)
        if normalize:
            tensor = tensor.to(torch.float32)/255
        return tensor

    def extract_feats(self, img1: torch.Tensor, img2: torch.Tensor):
        with torch.inference_mode():
            inp = torch.cat([img1, img2], dim=0)
            features1, features2 = self.disk(inp, self.num_features, pad_if_not_divisible=True)
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors
        return kps1, descs1, kps2, descs2

    def match(self, img1: torch.Tensor, img2: torch.Tensor)->MatchResult:

        # using these raises a warning for no benefits in matching apparently
        # hw1 = torch.tensor(img1.shape[2:], device=self.device)
        # hw2 = torch.tensor(img2.shape[2:], device=self.device)

        kps1, descs1, kps2, descs2 = self.extract_feats(img1, img2)

        with torch.inference_mode():

            lafs1 = KF.laf_from_center_scale_ori(kps1[None])
            lafs2 = KF.laf_from_center_scale_ori(kps2[None])

            dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2)

            cv2_matches = cv2_matches_from_kornia(dists, idxs)

            kps1 = opencv_kpts_from_laf(lafs1)
            kps2 = opencv_kpts_from_laf(lafs2)
            match_res = MatchResult(kps1, kps2, cv2_matches)
        return match_res
