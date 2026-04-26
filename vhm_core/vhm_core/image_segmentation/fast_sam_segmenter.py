from typing import List, Dict, Any, Optional

import gc
import torch
import numpy as np
import cv2
from ultralytics import FastSAM


class FastSAMSegmenter:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = 0.4,
        iou: float = 0.9,
        imgsz: int = 1024,
        retina_masks: bool = True,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.95):

        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.retina_masks = retina_masks
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

        self.model = FastSAM(self.model_path)

        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        self.warmup()
    
    def warmup(self, width: int = 640, height: int = 480, runs: int = 2):
        dummy = np.zeros((height, width, 3), dtype=np.uint8)

        for _ in range(runs):
            with torch.inference_mode():
                _ = self.model(
                    source=dummy,
                    device=self.device,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    retina_masks=self.retina_masks,
                    verbose=False,
                    half=True if self.device == "cuda" else False,
                )

        self.cleanup_gpu_memory()

    def segment_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if image is None:
            return []

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if image.ndim != 3:
            raise ValueError("image must be BGR/RGB with shape HxWx3")

        h, w = image.shape[:2]
        image_area = h * w

        with torch.inference_mode():
            results = self.model(
                source=image,
                device=self.device,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                retina_masks=self.retina_masks,
                verbose=False,
                half=True if self.device == "cuda" else False,
            )

        if not results:
            return []

        result = results[0]

        if result.masks is None:
            return []

        masks = result.masks.data.detach().cpu().numpy()

        boxes = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()

        # Not used
        # scores = []
        # if result.boxes is not None and result.boxes.conf is not None:
        #     scores = result.boxes.conf.detach().cpu().numpy()

        detections = []

        for idx, mask in enumerate(masks):
            mask_uint8 = self._to_uint8_mask(mask, target_size=(w, h))

            area = int(np.count_nonzero(mask_uint8))
            area_ratio = area / image_area if image_area > 0 else 0.0

            if area_ratio < self.min_area_ratio:
                continue

            if area_ratio > self.max_area_ratio:
                continue

            if idx < len(boxes):
                x1, y1, x2, y2 = boxes[idx].astype(int)
                bbox = self._clip_bbox([x1, y1, x2, y2], w, h)
            else:
                bbox = self._bbox_from_mask(mask_uint8)

            crop = self._crop_from_bbox(image, bbox)

            #score = float(scores[idx]) if idx < len(scores) else 0.0

            detections.append({
                "index": len(detections),
                "fast_sam_index": idx,
                "mask": mask_uint8,
                "bbox": bbox,
                "area": area,
                "area_ratio": round(area_ratio, 6),
                #"score": round(score, 6),
                "crop": crop,
            })

        return detections

    def _to_uint8_mask(
        self,
        mask: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        target_w, target_h = target_size

        if mask_uint8.shape[:2] != (target_h, target_w):
            mask_uint8 = cv2.resize(
                mask_uint8,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask_uint8

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> list[int]:
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]

        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]

    @staticmethod
    def _clip_bbox(bbox: list[int], width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(x1), width - 1))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height - 1))
        y2 = max(0, min(int(y2), height))

        return [x1, y1, x2, y2]

    @staticmethod
    def _crop_from_bbox(
        image: np.ndarray,
        bbox: list[int],
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    def cleanup_gpu_memory(self):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def unload_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        self.cleanup_gpu_memory()