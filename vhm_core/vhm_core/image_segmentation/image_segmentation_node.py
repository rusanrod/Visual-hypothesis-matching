import time
from pathlib import Path
import json

import cv2
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vhm_interfaces.srv import SegmentImage #type: ignore

from vhm_core.image_segmentation.fast_sam_segmenter import FastSAMSegmenter
from vhm_core.image_segmentation.mask_utils import save_mask, save_crop

from vhm_core.utlis.result_utils import VHMResultsManager

class FastSAMNode(Node):
    def __init__(self):
        super().__init__("image_segmentation_node")

        self.bridge = CvBridge()
        self.last_image_msg = None

        params = [
                ("image_topic", "/head_rgbd_sensor/rgb/image_rect_color"),
                ("model_path", "FastSAM-x.pt"),
                ("device", "cuda"),
                ("conf", 0.4),
                ("iou", 0.9),
                ("imgsz", 640),
                ("retina_masks", True),
            ]

        for name, default in params:
            self.declare_parameter(name, default)

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("image_topic").get_parameter_value().string_value,
            self.image_callback,
            10,
        )

        self.segmenter = FastSAMSegmenter(
            model_path=self.get_parameter("model_path").get_parameter_value().string_value,
            device=self.get_parameter("device").get_parameter_value().string_value,
            conf=self.get_parameter("conf").get_parameter_value().double_value,
            iou=self.get_parameter("iou").get_parameter_value().double_value,
            imgsz=self.get_parameter("imgsz").get_parameter_value().integer_value,
            retina_masks=self.get_parameter("retina_masks").get_parameter_value().bool_value,
        )

        self.segmentation_srv = self.create_service(
            SegmentImage,
            "vhm_core/segment_image",
            self.segment_callback,
        )

        self.get_logger().info("FAST SAM segmentation node ready.")

    # === Image loader ===
    def _load_request_images(self, input_dir: str, results_mgr: VHMResultsManager) -> list[dict]:
        input_dir = input_dir.strip()
        offline_mode = (len(input_dir) > 0)

        if offline_mode:
            return self._load_images_from_dir(input_dir, results_mgr)

        self.get_logger().info("No input_dir provided, using latest topic image.")
        return self._load_latest_topic_image()
    
    def _load_images_from_dir(self, input_dir: str, results_mgr: VHMResultsManager) -> list[dict]:
        input_path = Path(input_dir)

        if not input_path.exists():
            raise RuntimeError(f"input_dir does not exist: {input_path}")

        if not input_path.is_dir():
            raise RuntimeError(f"input_dir is not a directory: {input_path}")
        
        image_paths = results_mgr.collect_image_paths(input_dir)

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
            
            experiment_id = request.experiment_id or "test"
            results_mgr = VHMResultsManager(experiment_id=experiment_id)
            
            images = self._load_request_images(request.input_dir, results_mgr)

            if not images:
                raise RuntimeError("No images to process.")
            
            paths = {}
            if request.save_logs:
                paths = results_mgr.prepare_segmentation_dirs()

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

                    if request.save_logs:
                        mask_filepath = paths["masks_dir"] / f"{Path(image_name).stem}_mask_{seg_idx:03d}.png"
                        mask_path = save_mask(mask, mask_filepath)

                        crop_filepath = paths["crops_dir"] / f"{Path(image_name).stem}_crop_{seg_idx:03d}.png"
                        crop_path = save_crop(image, mask, bbox, crop_filepath)

                    image_record["segments"].append({
                        "segment_index": seg_idx,
                        "fast_sam_index": det["fast_sam_index"],
                        "bbox": bbox,
                        "area": det["area"],
                        "area_ratio": det["area_ratio"],
                        #"score": det["score"],
                        "mask_path": str(mask_path) if mask_path else "",
                        "crop_path": str(crop_path) if crop_path else "",
                    })

                segmentation_records.append(image_record)

            if request.save_logs:
                self._save_segmentation_info(
                    results_mgr=results_mgr,
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
        results_mgr: VHMResultsManager,
        segments: list[dict],
        start_time: float,
        end_time: float,
    ) -> str:
        
        output_path = results_mgr.segmentation_info_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_time = end_time - start_time
        processed_count = len(segments)
        total_segments = sum(len(item.get("segments", [])) for item in segments)

        payload = {
            "output_dir": str(results_mgr.segmentation_dir),
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
                "masks_dir": str(results_mgr.masks_dir),
                "crops_dir": str(results_mgr.crops_dir),
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