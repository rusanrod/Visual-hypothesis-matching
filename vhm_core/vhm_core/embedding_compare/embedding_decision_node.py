import json
import time
import torch
import rclpy
from rclpy.node import Node

from vhm_interfaces.srv import DecideEmbeddingMatch  # type: ignore


class EmbeddingDecisionNode(Node):
    def __init__(self):
        super().__init__("embedding_decision_node")

        self.decision_srv = self.create_service(
            DecideEmbeddingMatch,
            "vhm_core/decide_embedding_match",
            self.decide_callback,
        )

        self.get_logger().info("Embedding decision node ready.")

    def decide_callback(self, request, response):
        try:
            crop_count = int(request.crop_count)
            reference_count = int(request.reference_count)

            if crop_count <= 0 or reference_count <= 0:
                raise RuntimeError("Invalid similarity matrix dimensions.")

            expected_size = crop_count * reference_count
            if len(request.similarity_matrix) != expected_size:
                raise RuntimeError(
                    f"Invalid similarity matrix size. "
                    f"Expected {expected_size}, got {len(request.similarity_matrix)}"
                )

            similarity = torch.tensor(
                request.similarity_matrix,
                dtype=torch.float32,
            ).reshape(crop_count, reference_count)

            result = self._compute_reference_fusion(
                similarity=similarity,
                top_k=request.top_k,
                top_m=request.top_m,
                threshold=request.threshold,
                alpha=request.alpha,
                beta=request.beta,
                gamma=request.gamma,
                rrf_k=request.rrf_k,
            )

            best_list_index = result["best_crop_index"]

            if request.crop_file_indices:
                best_file_index = int(request.crop_file_indices[best_list_index])
            else:
                best_file_index = best_list_index

            if request.crop_names:
                best_crop_name = request.crop_names[best_list_index]
            else:
                best_crop_name = f"crop_{best_file_index:03d}"

            response.success = True
            response.message = "Embedding decision completed."

            response.best_crop_list_index = best_list_index
            response.best_crop_file_index = best_file_index
            response.best_crop_name = best_crop_name

            response.best_score = float(result["best_score"])
            response.best_raw_score = float(result["best_raw_score"])
            response.best_mean_top_m_score = float(result["best_mean_top_m_score"])
            response.best_rrf_score = float(result["best_rrf_score"])
            response.best_vote_count = int(result["best_vote_count"])
            response.accepted = bool(result["accepted"])

            response.crop_vote_counts = result["crop_vote_counts"]
            response.crop_final_scores = result["crop_final_scores"]
            response.crop_max_raw_scores = result["crop_max_raw_scores"]
            response.crop_mean_top_m_scores = result["crop_mean_top_m_scores"]
            response.crop_rrf_scores = result["crop_rrf_scores"]

            response.debug_json = json.dumps(result, indent=2)

            return response

        except Exception as e:
            self.get_logger().error(f"Decision error: {e}")

            response.success = False
            response.message = str(e)
            response.best_crop_list_index = -1
            response.best_crop_file_index = -1
            response.best_crop_name = ""
            response.best_score = 0.0
            response.accepted = False
            response.crop_vote_counts = []
            response.crop_final_scores = []
            response.crop_max_raw_scores = []
            response.crop_mean_top_m_scores = []
            response.crop_rrf_scores = []
            response.debug_json = ""

            return response
        
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