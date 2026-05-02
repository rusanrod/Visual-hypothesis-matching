#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import json


class SceneResultsTemplateGenerator:
    def __init__(self, scene_plan_path: Path, output_path: Path):
        self.scene_plan_path = scene_plan_path
        self.output_path = output_path

    def load_scene_plan(self) -> dict:
        if not self.scene_plan_path.exists():
            raise FileNotFoundError(f"scene_plan.json not found: {self.scene_plan_path}")

        with open(self.scene_plan_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_dataset_objects(self, scene_plan: dict) -> list[dict]:
        objects = {}

        for scene in scene_plan["scenes"]:
            for obj in scene["objects"]:
                instance_id = obj["instance_id"]

                objects[instance_id] = {
                    "class_name": obj["class_name"],
                    "instance_id": obj["instance_id"],
                }

        return sorted(objects.values(), key=lambda x: x["instance_id"])

    def generate(self) -> dict:
        scene_plan = self.load_scene_plan()
        dataset_objects = self.get_all_dataset_objects(scene_plan)

        results = []

        for scene in scene_plan["scenes"]:
            present_objects = {
                obj["instance_id"]: obj
                for obj in scene["objects"]
            }

            for query_obj in dataset_objects:
                gt_obj = present_objects.get(query_obj["instance_id"])
                is_present = gt_obj is not None

                result = {
                    "scene_id": scene["scene_id"],
                    "image_file": scene["image_file"],

                    "query": {
                        "class_name": query_obj["class_name"],
                        "instance_id": query_obj["instance_id"],
                    },

                    "ground_truth": {
                        "is_present": is_present,
                        "placement_id": gt_obj["placement_id"] if is_present else None,
                        "bbox_xyxy": (
                            gt_obj["ground_truth"]["bbox_xyxy"]
                            if is_present else None
                        ),
                        "mask_file": (
                            gt_obj["ground_truth"]["mask_file"]
                            if is_present else None
                        ),
                    },

                    "prediction": {
                        "detected": None,
                        "predicted_class": None,
                        "predicted_instance": None,
                        "bbox_xyxy": None,
                        "mask_file": None,
                        "crop_file": None,
                        "similarity_score": None,
                    },

                    "evaluation": {
                        "outcome": None,
                        "correct": None,
                        "confused_with": None,
                    }
                }

                results.append(result)

        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source_plan": str(self.scene_plan_path),
                "total_scenes": len(scene_plan["scenes"]),
                "total_objects": len(dataset_objects),
                "total_queries": len(results),
                "description": (
                    "Template for evaluating all dataset objects against all scenes."
                )
            },
            "dataset_objects": dataset_objects,
            "results": results,
        }

    def save(self):
        output = self.generate()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(f"Template saved to: {self.output_path}")
        print(f"Total scenes: {output['metadata']['total_scenes']}")
        print(f"Total objects: {output['metadata']['total_objects']}")
        print(f"Total queries: {output['metadata']['total_queries']}")


def main():
    base_dir = Path.home() / "vhm_ws" / "src" / "vhm_results"

    scene_plan_path = base_dir / "scene_plan" / "scene_plan_42_25.json"
    output_path = base_dir / "scene_results" / "scene_results_template.json"

    generator = SceneResultsTemplateGenerator(
        scene_plan_path=scene_plan_path,
        output_path=output_path,
    )

    generator.save()


if __name__ == "__main__":
    main()