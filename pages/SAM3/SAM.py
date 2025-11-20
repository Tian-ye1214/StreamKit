import numpy as np
import warnings
import os
import sys

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

from pages.SAM3.SAMCode.model_builder import build_sam3_image_model
from pages.SAM3.SAMCode.model.sam3_image_processor import Sam3Processor
import cv2


class SAMSegment:
    def __init__(self):
        self.SAM_model = build_sam3_image_model(bpe_path='pages/SAM3/bpe_simple_vocab_16e6.txt.gz'
                                                , checkpoint_path='pages/ModelCheckpoint/SAM3/sam3.pt'
                                                , enable_inst_interactivity=True)
        self.processor = Sam3Processor(self.SAM_model)

    def show_mask(self, mask, color=None, image=None, box=None):
        if color is None:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = (mask > 127).astype(np.uint8)

        if image is not None:
            image = cv2.resize(image, (w, h))
            result = image.copy()
            mask_colored = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_area = (mask > 0).reshape(h, w, 1)
            result = np.where(mask_area,
                              result * (1 - color[3]) + mask_colored[:, :, :3] * 255 * color[3],
                              result)
        else:
            result = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        result = cv2.drawContours(result.astype(np.uint8), contours, -1, (255, 255, 255), thickness=2)

        if box is not None:
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[2]), int(box[3])
            cv2.rectangle(result, (x0, y0), (x1, y1), (0, 0, 255), 2)

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

    def concept_inference(self, image, text, confidence_threshold=0.2):
        inference_state = self.processor.set_image(image)
        self.processor.reset_all_prompts(inference_state)
        self.processor.set_confidence_threshold(confidence_threshold, inference_state)
        inference_state = self.processor.set_text_prompt(state=inference_state, prompt=text)
        return inference_state['masks']


    def point_and_box_inference(self, image, input_point=None, input_label=None, input_box=None):
        inference_state = self.processor.set_image(image)
        self.processor.reset_all_prompts(inference_state)
        masks, scores, _ = self.SAM_model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        return masks
