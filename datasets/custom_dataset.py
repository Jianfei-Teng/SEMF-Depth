from __future__ import absolute_import, division, print_function

import numpy as np
import PIL.Image as pil

from data_paths import image_path
from .mono_dataset import MonoDataset


class CustomDataset(MonoDataset):
    """Fixed left-camera sequences with exact temporal-neighbor lookup."""
    def __init__(self, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[1.1624285, 0, 0.4981235, 0],
                           [0, 1.5498625, 0.4937840, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (480, 640)
        self.side_map = {"l": "l", "2": "l"}

    def index_to_folder_and_frame_idx(self, index):
        folder, frame, side = self.filenames[index].split()
        return folder, int(frame), side

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        return color.transpose(pil.Transpose.FLIP_LEFT_RIGHT) if do_flip else color

    def get_image_path(self, folder, frame_index, side):
        return str(image_path(self.data_path, folder, frame_index, side, self.img_ext))

    def get_depth(self, folder, frame_index, side, do_flip):
        return None

