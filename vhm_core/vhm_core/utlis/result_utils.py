# results_utils.py

from pathlib import Path
from typing import Iterable


class VHMResultsManager:
    """
    Maneja la estructura de carpetas de vhm_results usando experiment_id
    como identificador absoluto del experimento.
    """

    def __init__(
        self,
        experiment_id: str,
        root_dir: str | Path = Path.home() / "vhm_ws" / "src" / "vhm_results",
    ):
        self.root_dir = Path(root_dir)
        self.experiment_id = experiment_id

        # Define las rutas base para cada tipo de resultado
        self.generated_references_root = self.root_dir / "generated_references"
        self.segmentations_root = self.root_dir / "image_segmentations"
        self.embeddings_root = self.root_dir / "image_embeddings"

        # Rutas para archivos de escena (no dependen de experiment_id)
        self.scene_images_root = self.root_dir / "scene_images"
        self.scene_plan_root = self.root_dir / "scene_plan"
        self.scene_results_root = self.root_dir / "scene_results"

        # Asegurar que las carpetas globales existan al instanciar
        for global_dir in [self.scene_images_root, self.scene_plan_root, self.scene_results_root]:
            global_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Validaciones y utilidades genéricas
    # ============================================================
    def validate_experiment_id(self, experiment_id: str) -> str:
            experiment_id = experiment_id.strip()
            if not experiment_id:
                raise ValueError("experiment_id cannot be empty.")
            if "/" in experiment_id or "\\" in experiment_id:
                raise ValueError(
                    f"Invalid experiment_id '{experiment_id}'. Do not use path separators."
                )
            return experiment_id

    def collect_image_paths(
        self,
        input_dir: str | Path,
        extensions: Iterable[str] | None = None,
    ) -> list[Path]:
        input_dir = Path(input_dir)
        
        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
        valid_exts = {ext.lower() for ext in extensions}

        if not input_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory or doesn't exist: {input_dir}")

        return sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

    # ============================================================
    # Propiedades dinámicas del Experimento Actual
    # ============================================================

    # --- 1. Referencias Generadas ---
    @property
    def reference_dir(self) -> Path:
        return self.generated_references_root / self.experiment_id

    @property
    def reference_images_dir(self) -> Path:
        return self.reference_dir / "images"

    @property
    def reference_bank_path(self) -> Path:
        return self.reference_dir / "reference_bank.json"

    @property
    def reference_embeddings_path(self) -> Path:
        return self.reference_dir / "reference_embeddings.pth"

    # --- 2. Segmentaciones ---
    @property
    def segmentation_dir(self) -> Path:
        return self.segmentations_root / self.experiment_id

    @property
    def crops_dir(self) -> Path:
        return self.segmentation_dir / "crops"

    @property
    def masks_dir(self) -> Path:
        return self.segmentation_dir / "masks"

    @property
    def segmentation_info_path(self) -> Path:
        return self.segmentation_dir / "segmentation_info.json"

    # --- 3. Embeddings ---
    @property
    def embedding_results_dir(self) -> Path:
        return self.embeddings_root / self.experiment_id

    @property
    def comparison_results_path(self) -> Path:
        return self.embedding_results_dir / "comparison_results.json"

    @property
    def similarity_matrix_path(self) -> Path:
        return self.embedding_results_dir / "similarity_matrix.npy"

    @property
    def comparison_summary_path(self) -> Path:
        return self.embedding_results_dir / "comparison_summary.json"
    
    # ============================================================
    # Preparadores de Directorios (Creación bajo demanda)
    # ============================================================

    def prepare_reference_dirs(self) -> dict[str, Path]:
        """Crea y devuelve las rutas de referencias para el experimento."""
        self.reference_images_dir.mkdir(parents=True, exist_ok=True)
        return {
            "reference_dir": self.reference_dir,
            "reference_images_dir": self.reference_images_dir,
            "reference_bank_path": self.reference_bank_path,
            "reference_embeddings_path": self.reference_embeddings_path,
        }

    def prepare_segmentation_dirs(self) -> dict[str, Path]:
        """Crea y devuelve las rutas de segmentación para el experimento."""
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        return {
            "segmentation_dir": self.segmentation_dir,
            "crops_dir": self.crops_dir,
            "masks_dir": self.masks_dir,
            "segmentation_info_path": self.segmentation_info_path,
        }

    def prepare_embedding_results_dir(self) -> dict[str, Path]:
        """Crea y devuelve las rutas de embeddings para el experimento."""
        self.embedding_results_dir.mkdir(parents=True, exist_ok=True)
        return {
            "embedding_results_dir": self.embedding_results_dir,
            "comparison_results_path": self.comparison_results_path,
            "similarity_matrix_path": self.similarity_matrix_path,
            "comparison_summary_path": self.comparison_summary_path,
        }

    # ============================================================
    # Métodos de validación de existencia (Require)
    # ============================================================

    def require_reference_embeddings(self) -> Path:
        if not self.reference_embeddings_path.exists():
            raise FileNotFoundError(f"Missing embeddings: {self.reference_embeddings_path}")
        return self.reference_embeddings_path

    def require_reference_images_dir(self) -> Path:
        if not self.reference_images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.reference_images_dir}")
        return self.reference_images_dir

    def require_crops_dir(self) -> Path:
        if not self.crops_dir.is_dir():
            raise FileNotFoundError(f"Missing crops directory: {self.crops_dir}")
        return self.crops_dir