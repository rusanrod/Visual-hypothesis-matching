#!/usr/bin/env python3

from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import json


@dataclass
class ObjectInstance:
    class_name: str
    instance_id: str
    total_required: int
    remaining: int


class ScenePlanGenerator:
    def __init__(
        self,
        classes: list[str],
        instances_per_class: int = 5,
        appearances_per_instance: int = 20,
        min_objects_per_scene: int = 6,
        max_objects_per_scene: int = 8,
        seed: int = 42,
    ):
        self.classes = classes
        self.instances_per_class = instances_per_class
        self.appearances_per_instance = appearances_per_instance
        self.min_objects_per_scene = min_objects_per_scene
        self.max_objects_per_scene = max_objects_per_scene
        self.seed = seed

        random.seed(self.seed)

        self.objects = self._create_objects()

    def _create_objects(self) -> list[ObjectInstance]:
        objects = []

        for class_name in self.classes:
            for idx in range(1, self.instances_per_class + 1):
                instance_id = f"{class_name}_{idx:02d}"

                objects.append(
                    ObjectInstance(
                        class_name=class_name,
                        instance_id=instance_id,
                        total_required=self.appearances_per_instance,
                        remaining=self.appearances_per_instance,
                    )
                )

        return objects

    def _remaining_objects(self) -> list[ObjectInstance]:
        return [obj for obj in self.objects if obj.remaining > 0]

    def _select_scene_objects(self, scene_size: int) -> list[ObjectInstance]:
        available = self._remaining_objects()

        # Ordenamos por apariciones restantes para favorecer balance
        available.sort(key=lambda obj: obj.remaining, reverse=True)

        # Tomamos un subconjunto de los más necesitados y lo aleatorizamos
        candidate_pool_size = min(len(available), scene_size * 4)
        candidate_pool = available[:candidate_pool_size]
        random.shuffle(candidate_pool)

        selected = []
        selected_instance_ids = set()
        used_classes = {}

        for obj in candidate_pool:
            if len(selected) >= scene_size:
                break

            # No repetir la misma instancia en la misma escena
            if obj.instance_id in selected_instance_ids:
                continue

            # Evitar demasiados objetos de la misma clase en una escena
            class_count = used_classes.get(obj.class_name, 0)

            if class_count >= 3:
                continue

            selected.append(obj)
            selected_instance_ids.add(obj.instance_id)
            used_classes[obj.class_name] = class_count + 1

        # Si faltaron objetos, rellenamos sin restricción fuerte de clase
        if len(selected) < scene_size:
            #selected_ids = {obj.instance_id for obj in selected}

            for obj in available:
                if len(selected) >= scene_size:
                    break

                if obj.instance_id in selected_instance_ids:
                    selected.append(obj)
                    selected_instance_ids.add(obj.instance_id)

        return selected

    def generate(self) -> dict:
        scenes = []
        scene_idx = 1

        while self._remaining_objects():
            remaining_total = sum(obj.remaining for obj in self.objects)

            scene_size = random.randint(
                self.min_objects_per_scene,
                self.max_objects_per_scene
            )

            scene_size = min(scene_size, remaining_total)

            selected_objects = self._select_scene_objects(scene_size)

            scene_objects = []

            for placement_idx, obj in enumerate(selected_objects, start=1):
                obj.remaining -= 1

                scene_objects.append({
                    "placement_id": f"obj_{placement_idx:02d}",
                    "class_name": obj.class_name,
                    "instance_id": obj.instance_id,

                    # Cheatsheet / ground truth manual
                    "ground_truth": {
                        "bbox_xyxy": None,
                        "mask_file": None,
                    },

                })

            scenes.append({
                "scene_id": f"scene_{scene_idx:03d}",
                "image_file": f"scene_{scene_idx:03d}.png",
                "scene_summary": list(obj.instance_id for obj in selected_objects),
                "num_objects": len(scene_objects),
                "objects": scene_objects,
            })

            scene_idx += 1

        return self._build_output(scenes)

    def _build_output(self, scenes: list[dict]) -> dict:
        summary = {}

        for obj in self.objects:
            summary[obj.instance_id] = {
                "class_name": obj.class_name,
                "expected_appearances": obj.total_required,
                "remaining_after_generation": obj.remaining,
            }

        return {
            "dataset_plan": {
                "created_at": datetime.now().isoformat(),
                "seed": self.seed,
                "classes": self.classes,
                "instances_per_class": self.instances_per_class,
                "appearances_per_instance": self.appearances_per_instance,
                "min_objects_per_scene": self.min_objects_per_scene,
                "max_objects_per_scene": self.max_objects_per_scene,
                "total_scenes": len(scenes),
                "total_object_placements": sum(scene["num_objects"] for scene in scenes),
            },
            "object_summary": summary,
            "scenes": scenes,
        }

    def save(self, output_path: Path):
        plan = self.generate()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=4, ensure_ascii=False)

        print(f"Scene plan saved to: {output_path}")
        print(f"Total scenes: {plan['dataset_plan']['total_scenes']}")
        print(f"Total object placements: {plan['dataset_plan']['total_object_placements']}")


def main():
    classes = [
        "bowl",
        "cup",
        "cutlery",
        "tennis_shoes",
    ]

    generator = ScenePlanGenerator(
        classes=classes,
        instances_per_class=5,
        appearances_per_instance=25,
        min_objects_per_scene=6,
        max_objects_per_scene=8,
        seed=42,
    )

    output_path = Path.home() / "vhm_ws" / "src" / "vhm_results" / "scene_plan" / f"scene_plan_{generator.seed}_{generator.appearances_per_instance}.json"

    generator.save(output_path)


if __name__ == "__main__":
    main()