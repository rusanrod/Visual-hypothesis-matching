import time
from pathlib import Path
import json

import cv2
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from vhm_interfaces.srv import LoadEmbeddingReferences, CompareEmbeddingCrops #type: ignore

from vhm_core.embedding_compare.clip_embedder import CLIPEmbedder
from vhm_core.utlis.result_utils import VHMResultsManager


class EmbeddingCompareNode(Node):
    def __init__(self):
        super().__init__("embedding_compare_node")

        self.bridge = CvBridge()

        self.active_experiment_id = ""
        self.reference_embeddings = None
        self.reference_metadata = []
        self.reference_source = ""
        self.embedding_dim = 0

        params = [
                ("model_name", "openai/clip-vit-base-patch32"),
                ("device", "cuda"),
                ("dtype", "float16"),
            ]

        for name, default in params:
            self.declare_parameter(name, default)
        
        model_name = self.get_parameter("model_name").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value
        dtype = self.get_parameter("dtype").get_parameter_value().string_value

        self.embedder = CLIPEmbedder(
            model_name=model_name,
            device=device,
            dtype=dtype,
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

        self.get_logger().info("Embedding compare node ready.")


    # === Callbacks ===

    def load_references_callback(self, request, response):
        try:
            experiment_id = request.experiment_id or "test"
            source_type = request.source_type.strip().lower()

            results_mgr = VHMResultsManager(experiment_id=experiment_id)
            experiment_saved = False

            if source_type == "pth":
                pth_path = results_mgr.require_reference_embeddings()

                payload = self._load_references_from_pth(pth_path)

                embeddings = payload["embeddings"]
                metadata = payload.get("metadata", [])
                experiment_id = payload.get("experiment_id", experiment_id)

            elif source_type == "image_dir":
                image_dir = results_mgr.require_reference_images_dir()

                embeddings, metadata = self._load_references_from_image_dir(image_dir, results_mgr)

                if request.save_experiment:
                    paths = results_mgr.prepare_reference_dirs()

                    self._save_reference_embeddings(
                        experiment_id=experiment_id,
                        embeddings=embeddings,
                        metadata=metadata,
                        source_type=source_type,
                        pth_path=paths["reference_embeddings_path"],
                    )
                    experiment_saved = True

            elif source_type == "images":
                embeddings, metadata = self._load_references_from_ros_images(
                    images=request.images,
                )

                if request.save_experiment:
                    paths = results_mgr.prepare_reference_dirs()

                    self._save_reference_embeddings(
                        experiment_id=experiment_id,
                        embeddings=embeddings,
                        metadata=metadata,
                        source_type=source_type,
                        pth_path=paths["reference_embeddings_path"],
                    )
                    experiment_saved = True

            else:
                raise RuntimeError(
                    f"Invalid source_type '{request.source_type}'. "
                    "Use: 'pth', 'image_dir', or 'images'."
                )
            
            embeddings = embeddings.detach().cpu()

            if embeddings.ndim != 2:
                raise RuntimeError(f"Embeddings must be a 2D tensor. Got shape: {embeddings.shape}")

            self.active_experiment_id = experiment_id
            self.reference_embeddings = embeddings
            self.reference_metadata = metadata
            self.reference_source = source_type
            self.embedding_dim = int(embeddings.shape[-1])

            response.success = True
            response.message = (
                f"References loaded from '{source_type}' "
                f"for experiment '{experiment_id}'."
            )
            response.reference_count = int(embeddings.shape[0])
            response.embedding_dim = self.embedding_dim
            response.experiment_saved = experiment_saved

            return response

        except Exception as e:
            self.get_logger().error(f"Load references error: {e}")

            response.success = False
            response.message = str(e)
            response.reference_count = 0
            response.embedding_dim = 0
            response.experiment_saved = False

            return response

        finally:
            self.embedder.cleanup_gpu_memory()

    def compare_crops_callback(self, request, response):
        try:
            if self.reference_embeddings is None:
                raise RuntimeError("No active reference embeddings loaded.")
            
            experiment_id = request.experiment_id or "test"
            source_type = request.source_type.strip().lower()

            results_mgr = VHMResultsManager(experiment_id=experiment_id)
            
            crop_names = []
            crop_paths = []
            crops = []

            if source_type == "images":
                crops = self._ros_images_to_cv(request.crops)
                crop_names = [
                    f"crop_{idx:03d}" 
                    for idx in range(len(crops))
                    ]
            
            elif source_type == "image_dir":
                crops_dir = results_mgr.require_crops_dir()
                crop_paths = results_mgr.collect_image_paths(crops_dir)

                if not crop_paths:
                    raise RuntimeError(f"No crop images found in: {crops_dir}")

                for path in crop_paths:
                    crop = cv2.imread(str(path))
                    if crop is None:
                        self.get_logger().warn(f"Could not read crop image: {path}")
                        continue
                    crops.append(crop)
                    crop_names.append(path.name)

            else:
                raise RuntimeError(
                    f"Invalid source_type '{request.source_type}'. "
                    "Use: 'image_dir' or 'images'."
                )
            
            if not crops:
                raise RuntimeError("No valid crops received.")

            crop_embeddings = self.embedder.encode_cv_images(crops).detach().cpu()
            self.reference_embeddings = self.reference_embeddings.detach().cpu()
            similarity = crop_embeddings @ self.reference_embeddings.T
            
            top_k = getattr(request, "top_k", 3)
            threshold = getattr(request, "threshold", 0.25)

            voting_result = self._compute_reference_fusion(
                similarity=similarity,
                top_k=top_k,
                threshold=threshold,
            )

            best_crop_index = voting_result["best_crop_index"]
            best_score = voting_result["best_score"]
            accepted = voting_result["accepted"]
            
            save_experiment = getattr(request, "save_experiment", False)
            if save_experiment:
                paths = results_mgr.prepare_embedding_results_dir()

                results_payload = {
                    "experiment_id": experiment_id,
                    "reference_source": self.reference_source,
                    "crop_source": source_type,
                    "crop_count": int(similarity.shape[0]),
                    "reference_count": int(similarity.shape[1]),
                    "embedding_dim": self.embedding_dim,


                    "vote_parameters": {
                        "alpha": voting_result["alpha"],
                        "beta": voting_result["beta"],
                        "gamma": voting_result["gamma"],
                        "rrf_k": voting_result["rrf_k"],
                        "top_k": voting_result["top_k"],
                        "top_m": voting_result["top_m"],

                        "threshold": float(threshold),
                    },
                    "results_summary": {
                        "best_crop_index": best_crop_index,
                        "best_score": best_score,
                        "best_crop_name": crop_names[best_crop_index] if crop_names else None,
                        "best_raw_score": voting_result["best_raw_score"],
                        "best_mean_top_m_score": voting_result["best_mean_top_m_score"],
                        "best_rrf_score": voting_result["best_rrf_score"],
                        "best_vote_count": voting_result["best_vote_count"],

                        "accepted": accepted,
                    },

                    "crop_vote_counts": voting_result["crop_vote_counts"],
                    #"crop_vote_scores": voting_result["crop_vote_scores"],
                    "reference_votes": voting_result["reference_votes"],

                    #"crop_names": crop_names,
                    #"crop_paths": [str(p) for p in crop_paths],
                    "reference_metadata": self.reference_metadata,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(paths["comparison_results_path"], "w", encoding="utf-8") as f:
                    json.dump(results_payload, f, indent=4, ensure_ascii=False)

                np.save(
                    paths["similarity_matrix_path"],
                    similarity.detach().cpu().numpy()
                )
    
                self.get_logger().info(
                    f"Comparison results saved: {paths['comparison_results_path']}"
                )


            response.success = True
            response.message = "Embedding comparison completed."

            response.crop_count = int(similarity.shape[0])
            response.reference_count = int(similarity.shape[1])

            if request.return_similarity_matrix:
                response.similarity_matrix = [
                    float(x) for x in similarity.flatten().tolist()
                ]
            else:
                response.similarity_matrix = []

            response.best_crop_index = best_crop_index
            response.best_score = best_score
            response.accepted = bool(accepted)
            response.crop_vote_counts = voting_result["crop_vote_counts"]
            #response.crop_vote_scores = voting_result["crop_vote_scores"]

            return response

        except Exception as e:
            self.get_logger().error(f"Compare crops error: {e}")

            response.success = False
            response.message = str(e)

            response.crop_count = 0
            response.reference_count = 0
            response.similarity_matrix = []

            response.best_crop_index = -1
            response.best_score = 0.0
            response.accepted = False
            response.crop_vote_counts = []
            #response.crop_vote_scores = []

            return response

        finally:
            self.embedder.cleanup_gpu_memory()

    # === Data load Methods ===
    def _load_references_from_pth(self, pth_path: Path) -> dict:
        if not pth_path.exists():
            raise RuntimeError(f"pth_path does not exist: {pth_path}")

        payload = torch.load(
            str(pth_path), 
            map_location="cpu", 
            weights_only=True
        )

        if "embeddings" not in payload:
            raise RuntimeError(f"Invalid cache file. Missing 'embeddings': {pth_path}")

        embeddings = payload["embeddings"]

        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("'embeddings' must be a torch.Tensor")

        return payload
    
    def _load_references_from_image_dir(
        self,
        image_dir: Path,
        results_mgr: VHMResultsManager,
    ) -> tuple[torch.Tensor, list[dict]]:

        if not image_dir.exists():
            raise RuntimeError(f"image_dir does not exist: {image_dir}")

        image_paths = results_mgr.collect_image_paths(image_dir)

        if not image_paths:
            raise RuntimeError(f"No reference images found in: {image_dir}")

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
    

    def _save_reference_embeddings(
        self,
        experiment_id: str,
        embeddings: torch.Tensor,
        metadata: list[dict],
        source_type: str,
        pth_path: Path,
    ) -> str:

        pth_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_id": experiment_id,
            "source_type": source_type,
            "model_name": self.embedder.model_name,
            "embedding_dim": int(embeddings.shape[-1]),
            "reference_count": int(embeddings.shape[0]),
            "embeddings": embeddings.detach().cpu(),
            "metadata": metadata,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        torch.save(payload, str(pth_path))
        self.get_logger().info(f"Reference embeddings saved: {pth_path}")

        return str(pth_path)
        
    def _compute_reference_fusion(
        self,
        similarity: torch.Tensor,
        top_k: int = 3,
        top_m: int = 3,
        threshold: float = 0.75,
        alpha: float = 0.60,
        beta: float = 0.30,
        gamma: float = 0.10,
        rrf_k: float = 60.0,
    ) -> dict:
        """
        similarity: [num_crops, num_references]

        Cada crop obtiene:
        - max_raw_score: mejor similitud directa contra cualquier referencia
        - mean_top_m_raw_score: promedio de sus mejores m similitudes
        - rrf_score: score por rankings de cada referencia
        - final_score: combinación ponderada
        """

        if similarity.ndim != 2:
            raise RuntimeError(
                f"Similarity matrix must be 2D. Got shape: {similarity.shape}"
            )

        num_crops, num_references = similarity.shape

        if num_crops == 0 or num_references == 0:
            raise RuntimeError("Empty similarity matrix.")

        top_k = max(1, min(top_k, num_crops))
        top_m = max(1, min(top_m, num_references))

        # 1) Score fuerte: mejor referencia por crop
        crop_max_raw_scores, crop_best_reference_indices = similarity.max(dim=1)

        # 2) Score estable: promedio de las mejores M referencias por crop
        crop_top_m_scores, _ = torch.topk(
            similarity,
            k=top_m,
            dim=1,
            largest=True,
        )
        crop_mean_top_m_scores = crop_top_m_scores.mean(dim=1)

        # 3) Ranking fusion: cada referencia produce ranking de crops
        rrf_scores = torch.zeros(
            num_crops,
            dtype=torch.float32,
            device=similarity.device,
        )

        crop_vote_counts = torch.zeros(
            num_crops,
            dtype=torch.int32,
            device=similarity.device,
        )

        reference_votes = []

        for ref_idx in range(num_references):
            ref_scores = similarity[:, ref_idx]

            ranked_crop_indices = torch.argsort(
                ref_scores,
                descending=True,
            )

            vote_items = []

            for rank, crop_idx_tensor in enumerate(ranked_crop_indices[:top_k]):
                crop_idx = int(crop_idx_tensor.item())
                raw_score = float(ref_scores[crop_idx].item())

                # rank empieza en 0; usamos rank + 1 para fórmula RRF
                rrf_increment = 1.0 / (rrf_k + rank + 1)

                rrf_scores[crop_idx] += rrf_increment
                crop_vote_counts[crop_idx] += 1

                vote_items.append({
                    "rank": int(rank),
                    "crop_index": crop_idx,
                    "raw_score": raw_score,
                    "rrf_increment": float(rrf_increment),
                })

            reference_votes.append({
                "reference_index": int(ref_idx),
                "votes": vote_items,
            })

        # Normalizamos RRF para que quede aprox en [0, 1]
        max_possible_rrf = num_references * (1.0 / (rrf_k + 1.0))
        if max_possible_rrf > 0:
            rrf_scores_norm = rrf_scores / max_possible_rrf
        else:
            rrf_scores_norm = rrf_scores

        # 4) Fusión final
        final_scores = (
            alpha * crop_max_raw_scores
            + beta * crop_mean_top_m_scores
            + gamma * rrf_scores_norm
        )

        best_crop_index = int(torch.argmax(final_scores).item())
        best_reference_index = int(crop_best_reference_indices[best_crop_index].item())

        best_final_score = float(final_scores[best_crop_index].item())
        best_raw_score = float(crop_max_raw_scores[best_crop_index].item())
        best_mean_top_m_score = float(crop_mean_top_m_scores[best_crop_index].item())
        best_rrf_score = float(rrf_scores_norm[best_crop_index].item())
        best_vote_count = int(crop_vote_counts[best_crop_index].item())

        accepted = best_final_score >= threshold

        return {
            "best_crop_index": best_crop_index,
            "best_reference_index": best_reference_index,
            "best_score": best_final_score,
            "best_raw_score": best_raw_score,
            "best_mean_top_m_score": best_mean_top_m_score,
            "best_rrf_score": best_rrf_score,
            "best_vote_count": best_vote_count,
            "accepted": bool(accepted),
            "threshold": float(threshold),
            "top_k": int(top_k),
            "top_m": int(top_m),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "rrf_k": float(rrf_k),
            "crop_final_scores": [
                float(x) for x in final_scores.detach().cpu().tolist()
            ],
            "crop_max_raw_scores": [
                float(x) for x in crop_max_raw_scores.detach().cpu().tolist()
            ],
            "crop_mean_top_m_scores": [
                float(x) for x in crop_mean_top_m_scores.detach().cpu().tolist()
            ],
            "crop_rrf_scores": [
                float(x) for x in rrf_scores_norm.detach().cpu().tolist()
            ],
            "crop_vote_counts": [
                int(x) for x in crop_vote_counts.detach().cpu().tolist()
            ],
            "reference_votes": reference_votes,
        }


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