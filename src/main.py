from EVbatteryDetection.vision_system.template_detector import TemplateDetector, DetectionResult
from EVbatteryDetection.utils.visualizer import *
from EVbatteryDetection.utils.data_loader import process_annotations, load_image

# TODO: add cli parse args

def main():
    # Load templates
    templates_dir = 'data/images/templates'
    matcher = TemplateDetector(*process_annotations(templates_dir))
    all_images_dir = 'data/images/all'
    all_imgs, _ = process_annotations(all_images_dir)

    for p in all_imgs.values():
        det_res: DetectionResult = matcher.one_templ_detection(p)
        if det_res is None:
            continue
        test_img = load_image(p, color=False)
        templ_img = cv2.cvtColor(det_res.templ_img, cv2.COLOR_RGB2GRAY)
        draw_matches(templ_img, det_res.match_res.kps1, test_img, det_res.match_res.kps2, det_res.match_res.matches, True)

        test_img_color = load_image(p, color=True)

        overlay_mask_with_border(test_img_color, mask=det_res.mask, polygon=det_res.polygon, draw_now=True)


if __name__ == "__main__":
    main()
