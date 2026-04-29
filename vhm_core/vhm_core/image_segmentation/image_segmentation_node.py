from pathlib import Path
import json
import time

import cv2
from matplotlib import image
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vhm_interfaces.srv import SegmentImage

from vhm_core.image_segmentation.fast_sam_segmenter import FastSAMSegmenter
from vhm_core.image_segmentation.mask_utils import save_mask, save_crop


class FastSAMNode(Node):
    def __init__(self):
        super().__init__("image_segmentation_node")

        self.bridge = CvBridge()
        self.last_image_msg = None

        self.declare_parameters(
            namespace="",
            parameters=[
                ("image_topic", "/hardware/camera/image_raw"),
                ("model_path", "FastSAM-x.pt"),
                ("device", "cuda"),
                ("conf", 0.4),
                ("iou", 0.9),
                ("imgsz", 640),
                ("retina_masks", True),
            ],
        )

        self.default_output_dir = Path.joinpath(Path.home(), "vhm_ws", "src", "vhm_results", "image_segmentations")

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            10,
        )

        self.segmenter = FastSAMSegmenter(
            model_path=self.get_parameter("model_path").value,
            device=self.get_parameter("device").value,
            conf=self.get_parameter("conf").value,
            iou=self.get_parameter("iou").value,
            imgsz=self.get_parameter("imgsz").value,
            retina_masks=self.get_parameter("retina_masks").value,
        )

        self.segmentation_srv = self.create_service(
            SegmentImage,
            "vhm_core/segment_image",
            self.segment_callback,
        )

        self.get_logger().info("FAST SAM segmentation node ready.")

    def _build_output_dir(self, reference_bank_id: str) -> Path:
        bank_id = reference_bank_id.strip() if reference_bank_id else "default"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.default_output_dir) / bank_id / timestamp

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


    # === Image loader ===
    def _load_request_images(self, input_dir: str) -> tuple[bool, list[dict]]:
        offline_mode = bool(input_dir.strip())

        if offline_mode:
            return offline_mode, self._load_images_from_dir(input_dir)

        return offline_mode, self._load_latest_topic_image()
    
    def _load_images_from_dir(self, input_dir: str) -> list[dict]:
        input_path = Path(input_dir)

        if not input_path.exists():
            raise RuntimeError(f"input_dir does not exist: {input_path}")

        if not input_path.is_dir():
            raise RuntimeError(f"input_dir is not a directory: {input_path}")

        # We look for common image file extensions
        exts = {".jpg", ".jpeg", ".png"}

        image_paths = [
            p for p in Path(input_dir).iterdir()
            if p.suffix.lower() in exts
        ]

        if not image_paths:
            raise RuntimeError(f"No images found in input_dir: {input_path}")

        images = []

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                self.get_logger().warn(f"Could not read image: {image_path}")
                continue

            images.append({
                "name": image_path.name,
                "image": image,
            })

        if not images:
            raise RuntimeError(f"No valid images found in input_dir: {input_path}")

        return images

    def _load_latest_topic_image(self) -> list[dict]:
        if self.last_image_msg is None:
            return []

        image = self.bridge.imgmsg_to_cv2(self.last_image_msg, desired_encoding="bgr8")

        return [{
            "name": f"topic_image_{int(time.time())}.png",
            "image": image,
        }]
    
    # === Callbacks ===
    
    def image_callback(self, msg: Image):
        self.last_image_msg = msg

    def segment_callback(self, request, response):
        start_time = time.time()

        try:

            all_masks_msg = []
            all_crops_msg = []
            segmentation_records = []
            
            offline_mode, images = self._load_request_images(request.input_dir)

            if not images:
                raise RuntimeError("No images to process.")
            
            output_dir: Path | None = None
            masks_dir: Path | None = None
            crops_dir: Path | None = None

            if offline_mode:
                output_dir = self._build_output_dir(request.reference_bank_id)

                masks_dir = output_dir / "masks"
                crops_dir = output_dir / "crops"

                if request.save_masks:
                    masks_dir.mkdir(parents=True, exist_ok=True)

                if request.save_crops:
                    crops_dir.mkdir(parents=True, exist_ok=True)
            
            for image_idx, item in enumerate(images):
                image = item["image"]
                image_name = item["name"]
                
                detections = self.segmenter.segment_image(image)

                image_record = {"segments": []}

                for det in detections:
                    seg_idx = det["index"]
                    mask = det["mask"]
                    bbox = det["bbox"]
                    crop = det["crop"]

                    mask_path = ""
                    crop_path = ""

                    mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
                    all_masks_msg.append(mask_msg)
                    crop_msg = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
                    all_crops_msg.append(crop_msg)

                    if offline_mode and masks_dir and crops_dir:
                        if request.save_masks:
                            if offline_mode:
                                mask_path = masks_dir / f"{Path(image_name).stem}_mask_{seg_idx:03d}.png"
                                mask_path = save_mask(mask, mask_path)

                        if request.save_crops:
                            if offline_mode:
                                crop_path = crops_dir / f"{Path(image_name).stem}_crop_{seg_idx:03d}.png"
                                crop_path = save_crop(image, mask, bbox, crop_path)


                    image_record["segments"].append({
                        "segment_index": seg_idx,
                        "fast_sam_index": det["fast_sam_index"],
                        "bbox": bbox,
                        "area": det["area"],
                        "area_ratio": det["area_ratio"],
                        #"score": det["score"],
                        "mask_path": mask_path,
                        "crop_path": crop_path,
                    })

                segmentation_records.append(image_record)

            if offline_mode:
                self._save_segmentation_info(
                    output_dir=output_dir,
                    segments=segmentation_records,
                    start_time=start_time,
                    end_time=time.time(),
                )

            response.success = True
            response.message = (
                f"Processed {len(images)} image(s), "
                f"{len(all_masks_msg)} mask(s), "
                f"{len(all_crops_msg)} crop(s)."
            )
            response.masks = all_masks_msg
            response.crops = all_crops_msg

            return response

        except Exception as e:
            self.get_logger().error(f"FAST SAM error: {e}")

            response.success = False
            response.message = str(e)
            response.masks = []
            response.crops = []

            return response

        finally:
            self.segmenter.cleanup_gpu_memory()


    def _save_segmentation_info(
        self,
        output_dir: Path,
        segments: list[dict],
        start_time: float,
        end_time: float,
    ) -> str:
        
        output_path = output_dir / "segmentation_info.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_time = end_time - start_time
        processed_count = len(segments)
        total_segments = sum(item.get("segment_count", 0) for item in segments)

        payload = {
            "output_dir": str(output_dir),

            "processed_image_count": processed_count,

            "total_segment_count": total_segments,

            "segmentation_started_at": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(start_time),
            ),
            "segmentation_finished_at": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(end_time),
            ),
            "total_segmentation_time_sec": round(total_time, 4),
            "avg_time_per_image_sec": round(
                total_time / processed_count,
                4,
            ) if processed_count > 0 else 0.0,
        
            "model": {
                "model_path": self.get_parameter("model_path").value,
                "device": self.segmenter.device,
                "conf": self.segmenter.conf,
                "iou": self.segmenter.iou,
                "imgsz": self.segmenter.imgsz,
            },

            "outputs": {
                "masks_dir": str(output_dir / "masks"),
                "crops_dir": str(output_dir / "crops"),
            },

            "images": segments,
        }


        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(output_path)


def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()