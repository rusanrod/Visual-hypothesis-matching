#!/usr/bin/env python3
import json
import re
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


class CropAnnotator:
    def __init__(self, plan_path, crops_dir):
        self.plan_path = Path(plan_path)
        self.crops_dir = Path(crops_dir)

        self.plan = self.load_json(self.plan_path)
        self.crops = self.load_crops()

        self.current_idx = 0
        self.selected_instance_id = None

        self.root = tk.Tk()
        self.root.title("VHM Crop Annotator")

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)

        self.info_label = tk.Label(self.root, font=("Arial", 14))
        self.info_label.pack(pady=5)

        self.buttons_frame = tk.Frame(self.root)
        self.buttons_frame.pack(pady=10)

        self.status_label = tk.Label(self.root, font=("Arial", 11))
        self.status_label.pack(pady=5)

        self.root.bind("<Return>", self.next_crop)

        self.root.bind("<Right>", self.next_crop)
        self.root.bind("<Left>", self.prev_crop)

        self.root.bind("<Down>", self.next_scene)
        self.root.bind("<Up>", self.prev_scene)

        self.show_crop()

    def load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self):
        with open(self.plan_path, "w", encoding="utf-8") as f:
            json.dump(self.plan, f, indent=4, ensure_ascii=False)

    def load_crops(self):
        exts = {".png", ".jpg", ".jpeg"}
        crops = []

        for path in sorted(self.crops_dir.iterdir()):
            if path.suffix.lower() not in exts:
                continue

            scene_id, crop_index = self.parse_crop_name(path.name)
            if scene_id is None:
                continue

            crops.append({
                "path": path,
                "scene_id": scene_id,
                "crop_index": crop_index,
            })

        if not crops:
            raise RuntimeError(f"No crops found in {self.crops_dir}")

        return crops

    def parse_crop_name(self, filename):
        """
        Espera nombres tipo:
        scene_001_crop_000.png
        scene_001_crop_014.jpg
        """

        match = re.search(r"(scene_\d+).*?crop_(\d+)", filename)

        if not match:
            return None, None

        scene_id = match.group(1)
        crop_index = int(match.group(2))

        return scene_id, crop_index

    def get_scene(self, scene_id):
        for scene in self.plan["scenes"]:
            if scene["scene_id"] == scene_id:
                return scene
        return None

    def get_objects_for_scene(self, scene_id):
        scene = self.get_scene(scene_id)
        if scene is None:
            return []
        return scene.get("objects", [])

    def annotate_object(self, instance_id):
        crop = self.crops[self.current_idx]
        objects = self.get_objects_for_scene(crop["scene_id"])

        for obj in objects:
            if obj["instance_id"] == instance_id:
                obj["ground_truth"]["crop_index"] = crop["crop_index"]
                obj["ground_truth"]["annotated"] = True
                #self.selected_instance_id = instance_id
                self.save_json()

                self.status_label.config(
                    text=f"Guardado: {instance_id} -> crop {crop['crop_index']:03d}"
                )
                self.refresh_buttons(crop["scene_id"])

                self.next_crop()
                return

    def skip_if_already_used(self):
        """
        Salta crops que ya estén asignados como ground_truth en su escena.
        """
        while self.current_idx < len(self.crops):
            crop = self.crops[self.current_idx]
            objects = self.get_objects_for_scene(crop["scene_id"])

            already_used = any(
                obj.get("ground_truth", {}).get("annotated") is True
                and obj.get("ground_truth", {}).get("crop_index") == crop["crop_index"]
                for obj in objects
            )

            if not already_used:
                break

            self.current_idx += 1

    def show_crop(self):
        self.skip_if_already_used()

        if self.current_idx >= len(self.crops):
            messagebox.showinfo("Listo", "Ya no hay crops pendientes.")
            self.root.quit()
            return

        crop = self.crops[self.current_idx]
        img = Image.open(crop["path"]).convert("RGB")
        img.thumbnail((700, 500))

        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)

        self.info_label.config(
            text=f"{crop['scene_id']} | crop_index: {crop['crop_index']:03d} | {crop['path'].name}"
        )

        self.refresh_buttons(crop["scene_id"])

    def refresh_buttons(self, scene_id):
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()

        objects = self.get_objects_for_scene(scene_id)

        pending_objects = [
            obj for obj in objects
            if not obj.get("ground_truth", {}).get("annotated", False)
        ]

        total = len(objects)
        pending = len(pending_objects)
        self.status_label.config(text=f"Objetos pendientes en {scene_id}: {pending} / {total}")
        
        if not pending_objects:
            tk.Label(
                self.buttons_frame,
                text=f"Escena completa: {scene_id}",
                fg="green",
                font=("Arial", 12, "bold")
            ).grid(row=0, column=0)
            return

        cols = 4

        for i,obj in enumerate(pending_objects):
            instance_id = obj["instance_id"]

            #annotated = obj["ground_truth"].get("annotated", False)
            #crop_index = obj["ground_truth"].get("crop_index", None)

            """            label = instance_id
            if annotated:
                label += f" ✓ ({crop_index})"""

            btn = tk.Button(
                self.buttons_frame,
                text=instance_id,
                width=24,
                height=2,
                command=lambda iid=instance_id: self.annotate_object(iid)
            )
            row = i // cols
            col = i % cols
            btn.grid(row=row, column=col, padx=4, pady=4)
            #btn.pack(side=tk.LEFT, padx=4, pady=4)

    def next_crop(self, event=None):
        self.current_idx += 1
        self.selected_instance_id = None
        self.show_crop()

    def prev_crop(self, event=None):
        self.current_idx = max(0, self.current_idx - 1)
        self.selected_instance_id = None
        self.show_crop()

    def get_current_scene_id(self):
        return self.crops[self.current_idx]["scene_id"]

    def jump_to_scene(self, target_scene_id):
        for i, crop in enumerate(self.crops):
            if crop["scene_id"] == target_scene_id:
                self.current_idx = i
                self.show_crop()
                return

    def get_scene_ids_from_crops(self):
        scene_ids = []
        for crop in self.crops:
            if crop["scene_id"] not in scene_ids:
                scene_ids.append(crop["scene_id"])
        return scene_ids

    def next_scene(self, event=None):
        scene_ids = self.get_scene_ids_from_crops()
        current_scene = self.get_current_scene_id()

        idx = scene_ids.index(current_scene)

        if idx < len(scene_ids) - 1:
            self.jump_to_scene(scene_ids[idx + 1])

    def prev_scene(self, event=None):
        scene_ids = self.get_scene_ids_from_crops()
        current_scene = self.get_current_scene_id()

        idx = scene_ids.index(current_scene)

        if idx > 0:
            self.jump_to_scene(scene_ids[idx - 1])

    def run(self):
        self.root.mainloop()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Path al scene_plan.json")
    parser.add_argument("--crops-dir", required=True, help="Carpeta de crops")
    args = parser.parse_args()

    app = CropAnnotator(args.plan, args.crops_dir)
    app.run()


if __name__ == "__main__":
    main()