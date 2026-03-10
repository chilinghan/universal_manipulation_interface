from typing import Callable, Optional, Sequence

import numpy as np

from umi.common.cv_util import draw_predefined_mask, get_image_transform, inpaint_tag

count = 0

def make_umi_image_processor(
    in_res: Sequence[int],
    out_res: Sequence[int],
    no_mirror: bool,
    mirror_swap: bool,
    fisheye_converter=None,
) -> Callable[[np.ndarray, Optional[dict]], np.ndarray]:
    """Build UMI image preprocessing used by SLAM replay-buffer generation.

    Processing order is kept consistent with scripts_slam_pipeline/07_generate_replay_buffer.py:
    1) optional tag inpaint (if detections are provided)
    2) predefined mask (gripper always, mirror optional)
    3) resize or fisheye rectification
    4) optional mirror swap
    """
    iw, ih = int(in_res[0]), int(in_res[1])
    ow, oh = int(out_res[0]), int(out_res[1])

    resize_tf = get_image_transform(in_res=(iw, ih), out_res=(ow, oh))

    is_mirror = None
    if mirror_swap:
        mirror_mask = np.ones((oh, ow, 3), dtype=np.uint8)
        mirror_mask = draw_predefined_mask(
            mirror_mask,
            color=(0, 0, 0),
            mirror=True,
            gripper=False,
            finger=False,
        )
        is_mirror = mirror_mask[..., 0] == 0

    def process(rgb_img: np.ndarray, tag_detection: Optional[dict] = None) -> np.ndarray:
        if rgb_img.shape != (ih, iw, 3):
            raise ValueError(f"Expected frame shape {(ih, iw, 3)}, got {rgb_img.shape}")

        img = rgb_img.copy()

        if tag_detection is not None:
            tag_dict = tag_detection.get("tag_dict", {})
            for det in tag_dict.values():
                corners = det.get("corners", None)
                if corners is None:
                    continue
                img = inpaint_tag(img, np.asarray(corners))

        img = draw_predefined_mask(
            img,
            color=(0, 0, 0),
            mirror=no_mirror,
            gripper=True,
            finger=False,
        )

        if fisheye_converter is None:
            img = resize_tf(img)
        else:
            img = fisheye_converter.forward(img)

        if mirror_swap:
            img[is_mirror] = img[:, ::-1, :][is_mirror]

        return img

    return process
