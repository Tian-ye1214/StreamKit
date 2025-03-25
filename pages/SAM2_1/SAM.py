import numpy as np
import torch
from pages.SAM2_1.sam2.build_sam import build_sam2
from pages.SAM2_1.sam2.sam2_image_predictor import SAM2ImagePredictor
from pages.SAM2_1.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2

class SAM2Segment:
    def __init__(self, sam2_checkpoint, model_cfg):
        self.device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        np.random.seed(3)
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        self.mask_generator = SAM2AutomaticMaskGenerator(build_sam2(self.model_cfg, self.sam2_checkpoint
                                                                    , device=self.device, apply_postprocessing=False))

    def show_mask(self, mask, color=None, image=None):
        if color is None:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = (mask > 127).astype(np.uint8)

        if image is not None:
            image = cv2.resize(image, (w, h))
            result = image.copy()
            mask_colored = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_area = (mask > 0).reshape(h, w, 1)
            result = np.where(mask_area,
                             result * (1 - color[3]) + mask_colored[:,:,:3] * 255 * color[3],
                             result)
        else:
            result = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        result = cv2.drawContours(result.astype(np.uint8), contours, -1, (255, 255, 255), thickness=2)

        return result

    def show_points(self, image, clicks):
        if not clicks:
            return image
        result = image.copy()
        for click in clicks:
            x, y = click["x"], click["y"]
            marker = click["marker"]

            color = (76, 175, 80) if marker == 1 else (244, 67, 54)
            border_color = (56, 142, 60) if marker == 1 else (211, 47, 47)

            cv2.circle(result, (x, y), 6, border_color, 2)
            cv2.circle(result, (x, y), 4, color, -1)

        return result

    def point_inference(self, image, input_point=None, input_label=None):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        return masks

    def auto_mask_genarator(self, image):
        masks = self.mask_generator.generate(image)
        return masks

    def show_masks(self, image, masks):
        """生成带有随机颜色和轮廓的掩码叠加图"""
        if len(masks) == 0:
            return np.zeros_like(image)

        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        h, w = image.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        
        for ann in sorted_masks:
            mask = ann['segmentation'].astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) 
            color = np.concatenate([np.random.random(3)[::-1], [0.8]])
            overlay[mask.astype(bool)] = color

            contours, _ = cv2.findContours(mask,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, 0.01, closed=True) for c in contours]
            cv2.drawContours(overlay, contours, -1, (0, 0, 1.0, 0.4), 1)
        return (overlay * 255).astype(np.uint8)

