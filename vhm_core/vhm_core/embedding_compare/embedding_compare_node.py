import time
from pathlib import Path

import torch
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from vhm_interfaces.srv import LoadEmbeddingReferences, CompareEmbeddingCrops

from vhm_core.embedding_compare.clip_embedder import CLIPEmbedder


class EmbeddingCompareNode(Node):
    def __init__(self):
        super().__init__("embedding_compare_node")

        self.bridge = CvBridge()

        self.active_reference_id = ""
        self.reference_embeddings = None
        self.reference_metadata = []
        self.reference_source = ""
        self.reference_cache_path = ""
        self.embedding_dim = 0

        self.declare_parameters(
            namespace="",
            parameters=[
                ("model_name", "openai/clip-vit-base-patch32"),
                ("device", "cuda"),
                ("dtype", "float16"),
            ],
        )

        self.embedder = CLIPEmbedder(
            model_name=self.get_parameter("model_name").value,
            device=self.get_parameter("device").value,
            dtype=self.get_parameter("dtype").value,
        )

        self.load_references_srv = self.create_service(
            LoadEmbeddingReferences,
            "vhm_core/load_embedding_references",
            self.load_references_callback,
            )

        self.compare_crops_srv = self.create_service(
            CompareEmbeddingCrops,
            "vhm_core/compare_embedding_crops",
            self.compare_crops_callback,
        )

        self.get_logger().info("embedding_compare_node ready.")

    # === Callbacks ===

    def load_references_callback(self, request, response):
        try:
            source_type = request.source_type.strip().lower()

            if source_type == "pth":
                payload = self._load_references_from_pth(request.pth_path)

                embeddings = payload["embeddings"]
                metadata = payload.get("metadata", [])
                reference_id = payload.get("reference_id", request.reference_id)
                cache_path = request.pth_path
                cache_saved = False

            elif source_type == "image_dir":
                embeddings, metadata = self._load_references_from_image_dir(
                    image_dir=request.image_dir,
                )

                reference_id = request.reference_id or Path(request.image_dir).name
                cache_path = ""
                cache_saved = False

                if request.save_cache:
                    cache_path = self._save_reference_cache(
                        reference_id=reference_id,
                        embeddings=embeddings,
                        metadata=metadata,
                        source_type=source_type,
                        output_dir=request.cache_output_dir or request.image_dir,
                    )
                    cache_saved = True

            elif source_type == "images":
                embeddings, metadata = self._load_references_from_ros_images(
                    images=request.images,
                )

                reference_id = request.reference_id or f"runtime_refs_{int(time.time())}"
                cache_path = ""
                cache_saved = False

                if request.save_cache:
                    output_dir = request.cache_output_dir or "/tmp/vhm_embedding_cache"
                    cache_path = self._save_reference_cache(
                        reference_id=reference_id,
                        embeddings=embeddings,
                        metadata=metadata,
                        source_type=source_type,
                        output_dir=output_dir,
                    )
                    cache_saved = True

            else:
                raise RuntimeError(
                    f"Invalid source_type '{request.source_type}'. "
                    "Use: 'pth', 'image_dir', or 'images'."
                )

            self.active_reference_id = reference_id
            self.reference_embeddings = embeddings.cpu()
            self.reference_metadata = metadata
            self.reference_source = source_type
            self.reference_cache_path = cache_path
            self.embedding_dim = int(embeddings.shape[-1])

            response.success = True
            response.message = "References loaded successfully."
            response.active_reference_id = self.active_reference_id
            response.reference_count = int(embeddings.shape[0])
            response.embedding_dim = self.embedding_dim
            response.cache_saved = cache_saved
            response.cache_path = cache_path

            return response

        except Exception as e:
            self.get_logger().error(f"Load references error: {e}")

            response.success = False
            response.message = str(e)
            response.active_reference_id = self.active_reference_id
            response.reference_count = 0
            response.embedding_dim = 0
            response.cache_saved = False
            response.cache_path = ""

            return response

        finally:
            self.embedder.cleanup_gpu_memory()

    def compare_crops_callback(self, request, response):
        try:
            if self.reference_embeddings is None:
                raise RuntimeError("No active reference embeddings loaded.")

            crops = self._ros_images_to_cv(request.crops)

            if not crops:
                raise RuntimeError("No crops received.")

            crop_embeddings = self.embedder.encode_cv_images(crops).cpu()

            similarity = crop_embeddings @ self.reference_embeddings.T

            crop_best_scores, crop_best_ref_indices = similarity.max(dim=1)

            best_crop_index = int(crop_best_scores.argmax().item())
            best_reference_index = int(crop_best_ref_indices[best_crop_index].item())
            best_score = float(crop_best_scores[best_crop_index].item())

            response.success = True
            response.message = "Embedding comparison completed."
            response.active_reference_id = self.active_reference_id

            response.crop_count = int(similarity.shape[0])
            response.reference_count = int(similarity.shape[1])

            if request.return_similarity_matrix:
                response.similarity_matrix = [
                    float(x) for x in similarity.flatten().tolist()
                ]
            else:
                response.similarity_matrix = []

            if request.return_best_matches:
                response.best_reference_indices = [
                    int(x) for x in crop_best_ref_indices.tolist()
                ]
                response.best_reference_scores = [
                    float(x) for x in crop_best_scores.tolist()
                ]
            else:
                response.best_reference_indices = []
                response.best_reference_scores = []

            response.best_crop_index = best_crop_index
            response.best_reference_index = best_reference_index
            response.best_score = best_score

            return response

        except Exception as e:
            self.get_logger().error(f"Compare crops error: {e}")

            response.success = False
            response.message = str(e)
            response.active_reference_id = self.active_reference_id

            response.crop_count = 0
            response.reference_count = 0
            response.similarity_matrix = []
            response.best_reference_indices = []
            response.best_reference_scores = []

            response.best_crop_index = -1
            response.best_reference_index = -1
            response.best_score = 0.0

            return response

        finally:
            self.embedder.cleanup_gpu_memory()
    
    # === Helper Methods ===

    def _collect_image_paths(self, input_dir: Path) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

        return sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ])

    def _load_references_from_pth(self, pth_path: str) -> dict:
        path = Path(pth_path)

        if not path.exists():
            raise RuntimeError(f"pth_path does not exist: {path}")

        payload = torch.load(str(path), map_location="cpu")

        if "embeddings" not in payload:
            raise RuntimeError(f"Invalid cache file. Missing 'embeddings': {path}")

        embeddings = payload["embeddings"]

        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("'embeddings' must be a torch.Tensor")

        return payload
    
    def _load_references_from_image_dir(
        self,
        image_dir: str,
    ) -> tuple[torch.Tensor, list[dict]]:
        path = Path(image_dir)

        if not path.exists():
            raise RuntimeError(f"image_dir does not exist: {path}")

        if not path.is_dir():
            raise RuntimeError(f"image_dir is not a directory: {path}")

        image_paths = self._collect_image_paths(path)

        if not image_paths:
            raise RuntimeError(f"No reference images found in: {path}")

        embeddings = self.embedder.encode_image_paths([str(p) for p in image_paths])

        metadata = [
            {
                "index": idx,
                "source": "image_dir",
                "image_path": str(p),
                "name": p.name,
            }
            for idx, p in enumerate(image_paths)
        ]

        return embeddings, metadata
    
    def _load_references_from_ros_images(
        self,
        images: list[Image],
    ) -> tuple[torch.Tensor, list[dict]]:
        cv_images = self._ros_images_to_cv(images)

        if not cv_images:
            raise RuntimeError("No valid reference images received.")

        embeddings = self.embedder.encode_cv_images(cv_images)

        metadata = [
            {
                "index": idx,
                "source": "ros_image",
                "name": f"reference_image_{idx:03d}",
            }
            for idx in range(len(cv_images))
        ]

        return embeddings, metadata
    
    def _ros_images_to_cv(self, images: list[Image]) -> list[np.ndarray]:
        cv_images = []

        for msg in images:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )

                if cv_img is not None and cv_img.size > 0:
                    cv_images.append(cv_img)

            except Exception as e:
                self.get_logger().warn(f"Could not convert ROS image: {e}")

        return cv_images
    
    def _save_reference_cache(
        self,
        reference_id: str,
        embeddings: torch.Tensor,
        metadata: list[dict],
        source_type: str,
        output_dir: str,
    ) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cache_path = output_path / f"{reference_id}_reference_embeddings.pth"

        payload = {
            "reference_id": reference_id,
            "source_type": source_type,
            "model_name": self.embedder.model_name,
            "embedding_dim": int(embeddings.shape[-1]),
            "reference_count": int(embeddings.shape[0]),
            "embeddings": embeddings.cpu(),
            "metadata": metadata,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        torch.save(payload, str(cache_path))
        return str(cache_path)
    


def main(args=None):
    rclpy.init(args=args)
    node = EmbeddingCompareNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()