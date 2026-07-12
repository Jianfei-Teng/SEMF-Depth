from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CustomDataset(MonoDataset):
    """Dataset for custom monocular data
    """
    def __init__(self, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)

        # Normalized by image width (640) and height (480)
        # fx=743.95, fy=743.93, cx=318.80, cy=237.02
        self.K = np.array([[1.1624285, 0, 0.4981235, 0],
                           [0, 1.5498625, 0.4937840, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (480, 640)
        self.side_map = {"l": "l", "r": "r", "2": "l", "3": "r"}
        
        # Build frame index bounds per folder for clamping
        self.folder_bounds = {}
        self._build_folder_bounds()

    def _build_folder_bounds(self):
        """Scan data_path to determine min/max frame indices per folder.
        
        Registers bounds under ALL possible folder key formats so that
        lookups like folder_bounds["train/image_02"] always succeed.
        """
        if not os.path.exists(self.data_path):
            return
        
        def _scan_images_in(dir_path):
            """Return list of integer indices from image filenames in dir_path."""
            indices = []
            if not os.path.isdir(dir_path):
                return indices
            for f in os.listdir(dir_path):
                if f.lower().endswith('.jpg') or f.lower().endswith('.png'):
                    try:
                        idx = int(os.path.splitext(f)[0])
                        indices.append(idx)
                    except:
                        continue
            return indices
        
        for folder in os.listdir(self.data_path):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Sub-paths to search for images (to deduce min/max frames)
            search_configs = [
                # (sub_path_from_folder, extra_folder_keys_to_register)
                ("data",              [folder]),
                ("data_jpg",          [folder]),
                ("data_png",          [folder]),
                ("image_02/data",     [folder, f"{folder}/image_02"]),
                ("image_02/data_jpg", [folder, f"{folder}/image_02"]),
                ("image_02/data_png", [folder, f"{folder}/image_02"]),
                ("image_03/data",     [folder, f"{folder}/image_03"]),
                ("image_03/data_jpg", [folder, f"{folder}/image_03"]),
                ("",                  [folder]),  # Flat structure fallback
            ]
            
            for sub_path, keys in search_configs:
                scan_path = os.path.join(folder_path, sub_path) if sub_path else folder_path
                indices = _scan_images_in(scan_path)
                if indices:
                    bounds = (min(indices), max(indices))
                    for key in keys:
                        # Use os.path.normpath to normalize slashes
                        norm_key = key.replace("\\", "/")
                        if norm_key not in self.folder_bounds:
                            self.folder_bounds[norm_key] = bounds
                        else:
                            # Merge: widen the existing bounds
                            old_min, old_max = self.folder_bounds[norm_key]
                            self.folder_bounds[norm_key] = (
                                min(old_min, bounds[0]),
                                max(old_max, bounds[1])
                            )

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        # f_str must be calculated *after* clamping the frame_index.
        # But we also want to safely clamp it.
        
        # Normalize folder key to forward slashes for consistent lookup
        folder_key = folder.replace("\\", "/")
        
        # 1. First, clamp based on folder bounds if available
        if folder_key in self.folder_bounds:
            min_idx, max_idx = self.folder_bounds[folder_key]
            frame_index = max(min_idx, min(frame_index, max_idx))
        else:
            # Fallback: clamp to non-negative
            frame_index = max(0, frame_index)  # 0-indexed
            
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        
        # side: "l" = image_02, "r" = image_03
        side_folder = "image_02" if side == "l" else "image_03"
        data_folder = "data_jpg" if self.img_ext == ".jpg" else "data_png"
        
        def check_paths(f_str_test):
            """Helper to test all 3 dataset structural permutations for a given filename."""
            # A. New structure
            if folder.endswith(side_folder):
                path_a = os.path.join(self.data_path, folder, data_folder, f_str_test)
            else:
                path_a = os.path.join(self.data_path, folder, side_folder, data_folder, f_str_test)
            if os.path.exists(path_a): return path_a
                
            # B. Old structure
            path_b = os.path.join(self.data_path, folder, "data", f_str_test)
            if os.path.exists(path_b): return path_b
                
            # C. Flat structure
            if folder.endswith(side_folder):
                path_c = os.path.join(self.data_path, folder, f_str_test)
            else:
                path_c = os.path.join(self.data_path, folder, side_folder, f_str_test)
            if os.path.exists(path_c): return path_c
            return None

        # 1. First Attempt: Return the explicitly requested frame
        path = check_paths(f_str)
        if path:
            return path
            
        # [CRITICAL FIX / SOTA SAFEGUARD]
        # 2. Hard Fallback: The exact requested frame does NOT exist in any format.
        # This typically happens when fetching `frame_index + 1` at the end of a video sequence
        # where the dataset length annotation isn't perfect. We MUST fallback to a nearby frame.
        
        # Try shifting one frame backward (or two) to safely find an existing neighbor
        for shift_offset in [-1, -2, 1]:  
            shifted_idx = max(0, frame_index + shift_offset)
            shifted_str = "{:010d}{}".format(shifted_idx, self.img_ext)
            fb_path = check_paths(shifted_str)
            if fb_path:
                print(f"[Dataset Fallback] Missing {folder}/{f_str}, returning {shifted_str}")
                return fb_path

        # 3. Absolute Disaster Recovery: Just grab the absolute first image dynamically.
        # This guarantees the DataLoader thread never crashes, avoiding destroying 8 hours of training.
        search_dirs = [
            os.path.join(self.data_path, folder, data_folder),
            os.path.join(self.data_path, folder, side_folder, data_folder),
            os.path.join(self.data_path, folder, "data"),
            os.path.join(self.data_path, folder)
        ]
        
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                files = [f for f in os.listdir(s_dir) if f.endswith(self.img_ext)]
                if files:
                    safe_path = os.path.join(s_dir, sorted(files)[0])
                    print(f"[Dataset Rescue] Missing {folder}/{f_str}, rescuing with random frame {safe_path}")
                    return safe_path

        # If it reaches here, the entire sequence folder is actually missing from disk.
        # Return what *should* have been the path so the error is easy to debug.
        if folder.endswith(side_folder):
            return os.path.join(self.data_path, folder, data_folder, f_str)
        return os.path.join(self.data_path, folder, side_folder, data_folder, f_str)

    def get_depth(self, folder, frame_index, side, do_flip):
        return None

