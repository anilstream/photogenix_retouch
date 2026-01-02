# Standard library
import os
import sys

# Inject ComfyUI CPU flag before any ComfyUI imports
sys.argv.append("--cpu")

# Third-party
import torch

# Local application
from retouch_utils import (
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    get_value_at_index,
    import_custom_nodes,
    output_to_bytes,
)

# ---- ComfyUI bootstrap ----
add_comfyui_directory_to_sys_path()
add_extra_model_paths()
import_custom_nodes()

from nodes import NODE_CLASS_MAPPINGS


class NanoBananaMaskedInpaint(object):
    def __init__(self):
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimagemask = NODE_CLASS_MAPPINGS["LoadImageMask"]()


        self.inpaintcropimproved = NODE_CLASS_MAPPINGS["InpaintCropImproved"]()
        self.nanobananabasicnode = NODE_CLASS_MAPPINGS["NanoBananaBasicNode"]()
        self.inpaintstitchimproved = NODE_CLASS_MAPPINGS["InpaintStitchImproved"]()

    def run(self, image, mask, prompt):

        with torch.inference_mode():
            input_image = self.loadimage.load_image(
                image=image)
            input_mask = self.loadimagemask.load_image(
                image=mask,channel="red")

            inpaintcropimproved_18 = self.inpaintcropimproved.inpaint_crop(
                downscale_algorithm="bilinear",
                upscale_algorithm="bicubic",
                preresize=False,
                preresize_mode="ensure minimum resolution",
                preresize_min_width=1024,
                preresize_min_height=1024,
                preresize_max_width=16384,
                preresize_max_height=16384,
                mask_fill_holes=True,
                mask_expand_pixels=40,
                mask_invert=False,
                mask_blend_pixels=32,
                mask_hipass_filter=0.1,
                extend_for_outpainting=False,
                extend_up_factor=1,
                extend_down_factor=1,
                extend_left_factor=1,
                extend_right_factor=1,
                context_from_mask_extend_factor=1.2,
                output_resize_to_target_size=True,
                output_target_width=1024,
                output_target_height=1024,
                output_padding="32",
                image=get_value_at_index(input_image, 0),
                mask=get_value_at_index(input_mask, 0),
            )

            nanobananabasicnode_38 = self.nanobananabasicnode.run(
                gemini_key=os.getenv("GEMINI_KEY"),
                prompt=prompt,
                aspect_ratio="match_input_image",
                resolution="1K",
                output_format="png",
                safety_filter_level="block_only_high",
                image_input=get_value_at_index(inpaintcropimproved_18, 1),
            )

            inpaintstitchimproved_32 = self.inpaintstitchimproved.inpaint_stitch(
                stitcher=get_value_at_index(inpaintcropimproved_18, 0),
                inpainted_image=get_value_at_index(nanobananabasicnode_38, 0),
            )

            output_image = get_value_at_index(inpaintstitchimproved_32, 0)
            image = output_to_bytes(output_image)

        return image



if __name__ == "__main__":
    image="/home/anil/DEV/image1.png"
    mask="/home/anil/DEV/mask1.png"
    prompt="Remove the wooden stand with books and object inside at the right side. Everything else in the image exactly same, including all other people, garment, background, lighting, poses and facial features."
    masked_inpainter = NanoBananaMaskedInpaint()
    output = masked_inpainter.run(image, mask, prompt)
    with open("output1.png", "wb") as f:
        f.write(output)
