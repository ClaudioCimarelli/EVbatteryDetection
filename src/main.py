from EVbatteryDetection.vision_system.template_detector import TemplateDetector, DetectionResult
from EVbatteryDetection.utils.visualizer import *
from EVbatteryDetection.utils.data_loader import process_annotations, load_image

from time import perf_counter

import numpy as np

# TODO: add cli parse args

def main():
    # Load templates
    templates_dir = 'data/images/templates'
    matcher = TemplateDetector(*process_annotations(templates_dir), resize_factor=.5)
    all_images_dir = 'data/images/all'
    all_imgs, _ = process_annotations(all_images_dir)
    times = []
    for p in all_imgs.values():
        t1_start = perf_counter()
        det_res: DetectionResult = matcher.one_templ_detection(p)
        times.append(perf_counter() - t1_start)
        if det_res is None:
            print(f"Detection not found in time: {times[-1]:.2f} secs")
            continue
        print(f"Detection found at Template ID{det_res.template_id} in time: {times[-1]:.2f} secs!")
        # test_img = load_image(p, color=False)
        # templ_img = cv2.cvtColor(det_res.templ_img, cv2.COLOR_RGB2GRAY)
        # draw_matches(templ_img, det_res.match_res.kps1, test_img, det_res.match_res.kps2, det_res.match_res.matches, True)

        test_img_color = cv2.resize(load_image(p, color=False), det_res.img_shape, interpolation=cv2.INTER_AREA)
        overlay_mask_with_border(test_img_color, mask=det_res.mask, polygon=det_res.polygon, draw_now=True)

    times = np.array(times)
    print(f"Avg time per detection:{times.mean():.2f} secs")
    print(f"Median time per detection:{np.median(times):.2f} secs")
    print(f"Max time per detection:{times.max():.2f} secs")
    print(f"Min time per detection:{times.min():.2f} secs")

if __name__ == "__main__":
    main()
