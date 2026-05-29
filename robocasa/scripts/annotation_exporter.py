import cv2
import os
import shutil
import imageio
import hashlib
import json
import numpy as np
import zipfile


class AnnotationExporter:
    def __init__(
        self,
        annotation_dir,
        env,
    ):
        self.env = env
        self.annotation_dir = annotation_dir
        if os.path.exists(annotation_dir):
            shutil.rmtree(annotation_dir)
        os.makedirs(annotation_dir, exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "segmentation"), exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "segmentation_label"), exist_ok=True)

        self.rgb_writer = imageio.get_writer(os.path.join(annotation_dir, "rgb.mp4"), fps=20)
        self.depth_writer = imageio.get_writer(os.path.join(annotation_dir, "depth.mp4"), fps=20)
        self.segmentation_writer = imageio.get_writer(
            os.path.join(annotation_dir, "segmentation.mp4"), fps=20
        )

        weights_dir = os.path.join(annotation_dir, "..", "..", "cosmos_weights")
        inference = {
            "prompt": f"a robotic {env.get_ep_meta().get('lang', '')}.",
            "input_video_path": os.path.join(annotation_dir, "rgb.mp4"),
            "vis": {
                "control_weight": os.path.join(
                    weights_dir,
                    "example1",
                    "vis_weights.pt",
                ),
            },
            "edge": {
                "control_weight": os.path.join(
                    weights_dir,
                    "example1",
                    "edge_weights.pt",
                ),
            },
            "depth": {
                "control_weight": os.path.join(
                    weights_dir,
                    "example1",
                    "depth_weights.pt",
                ),
            },
            "seg": {
                "input_control": os.path.join(annotation_dir, "segmentation.mp4"),
                "control_weight": os.path.join(
                    weights_dir,
                    "example1",
                    "seg_weights.pt",
                ),
            },
        }
        with open(
            os.path.join(
                annotation_dir,
                "inference_cosmos_transfer1_robot_spatiotemporal_weights.json",
            ),
            "w",
        ) as f:
            json.dump(inference, f, indent=4)

    def add_record_before_step(
        self, frame_index, rgb_data, depth_data, segmentation_data, geom_id2name
    ):
        self.rgb_writer.append_data(rgb_data)
        cv2.imwrite(
            os.path.join(
                self.annotation_dir,
                "rgb",
                f"rgb_{frame_index:06d}.png",
            ),
            cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR),
        )

        d = np.clip((depth_data - 0.9) * 2550.0, 0, 255).astype(np.uint8)
        ddd = np.stack([d] * 3, axis=-1)
        self.depth_writer.append_data(ddd)

        def hash_color_batch(colors):
            # colors: [N, 4] uint8 array
            hashes = np.zeros((colors.shape[0], 4), dtype=np.uint8)
            for i, color in enumerate(colors):
                key = bytes(color)
                digest = hashlib.md5(key).digest()  # 16 bytes
                hashes[i] = np.frombuffer(digest[:4], dtype=np.uint8)
            return hashes

        def remap_rgba_array_fast(input_array):
            reshaped = input_array.reshape(-1, 4).astype(np.uint8)
            unique_colors, inverse_indices = np.unique(reshaped, axis=0, return_inverse=True)
            hashed_unique = hash_color_batch(unique_colors)
            # Vectorized mapping using inverse_indices
            hashed_flat = hashed_unique[inverse_indices]
            return hashed_flat.reshape(input_array.shape)

        ch0 = (segmentation_data[:, :, 1].astype(np.uint32) % 255).astype(np.uint8)
        ch1 = (segmentation_data[:, :, 1].astype(np.uint32) // 255 % 255).astype(np.uint8)
        ch2 = (segmentation_data[:, :, 1].astype(np.uint32) // 255 // 255 % 255).astype(np.uint8)
        ch3 = 255 - (segmentation_data[:, :, 1].astype(np.uint32) // 255 // 255 // 255).astype(
            np.uint8
        )
        rgba_frame = np.stack([ch0, ch1, ch2, ch3], axis=-1)
        rgba_frame = remap_rgba_array_fast(rgba_frame)
        bgra_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(
            os.path.join(
                self.annotation_dir,
                "segmentation",
                f"semantic_segmentation_{frame_index:06d}.png",
            ),
            bgra_frame,
        )
        self.segmentation_writer.append_data(rgba_frame)

        assert np.all(segmentation_data[:, :, 0] == 5)
        labels = {}
        mask = segmentation_data[:, :, 0] == 5
        ids = segmentation_data[:, :, 1][mask]
        unique_ids = np.unique(ids)
        for id in unique_ids:
            ch0 = (id % 255).astype(np.uint8)
            ch1 = (id // 255 % 255).astype(np.uint8)
            ch2 = (id // 255 // 255 % 255).astype(np.uint8)
            ch3 = 255 - (id // 255 // 255 // 255).astype(np.uint8)
            rgba = str(tuple(remap_rgba_array_fast(np.array([ch0, ch1, ch2, ch3]))))
            labels[rgba] = {"class": geom_id2name(id)}
        with open(
            os.path.join(
                self.annotation_dir,
                "segmentation_label",
                f"semantic_segmentation_labels_{frame_index:06d}.json",
            ),
            "w",
        ) as f:
            json.dump(labels, f, indent=4)

    def add_record_after_step(self):
        pass

    def finish(self):
        self.rgb_writer.close()
        self.depth_writer.close()
        self.segmentation_writer.close()

        # output the bounding box
        with open(
            os.path.join(
                self.annotation_dir,
                "name_mapping.json",
            ),
            "w",
        ) as f:
            name_mapping = {}
            for body_name in self.env.obj_body_id.keys():
                name_mapping[body_name] = self.env.get_obj_lang(obj_name=body_name)
            for id, name in self.env.sim.model._geom_id2name.items():
                if "distractor_fixture_back_edge_toaster_0" in name:
                    name_mapping["distractor_fixture_back_edge_toaster_0"] = "toaster"
                if "distractor_fixture_back_edge_plant_0" in name:
                    name_mapping["distractor_fixture_back_edge_plant_0"] = "plant"
            json.dump(name_mapping, f, indent=4)
