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

from PIL import Image

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
        self.prompt_suffix = "Everything else in the image exactly same, including all other people, garment, background, lighting, poses and facial features. Make sure inpainted region/object does not touch image border. Make sure inpainted object/region is not cropped by image border."

    def run(self, image, mask, prompt, resolution='1K',
            mask_expand_pixels=80, mask_blend_pixels=9, mask_hipass_filter=0.1, mask_fill_holes=False):

        # choose the processing resolution
        size = 1024 if resolution == "1K" else 2048 if resolution == "2K" else 4096
        prompt = prompt + self.prompt_suffix

        # verify the file paths
        if not os.path.exists(image):
            raise FileNotFoundError(f"Input image path does not exist: {image}")

        if not os.path.exists(mask):
            raise FileNotFoundError(f"Mask path does not exist: {mask}")

        print("IMAGE SIZE: ",Image.open(image).size)
        print("MASK SIZE: ", Image.open(mask).size)

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
                mask_fill_holes=mask_fill_holes,
                mask_expand_pixels=mask_expand_pixels,
                mask_invert=False,
                mask_blend_pixels=mask_blend_pixels,
                mask_hipass_filter=mask_hipass_filter,
                extend_for_outpainting=False,
                extend_up_factor=1,
                extend_down_factor=1,
                extend_left_factor=1,
                extend_right_factor=1,
                context_from_mask_extend_factor=1,
                output_resize_to_target_size=True,
                output_target_width=size,
                output_target_height=size,
                output_padding="32",
                image=get_value_at_index(input_image, 0),
                mask=get_value_at_index(input_mask, 0),
                device_mode="cpu (compatible)"
            )

            nanobananabasicnode_38 = self.nanobananabasicnode.run(
                gemini_key=os.getenv("GEMINI_KEY"),
                prompt=prompt,
                aspect_ratio="match_input_image",
                resolution=resolution,
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

class NanoBananaBgChange(object):

    def __init__(self):
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()


        self.getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
        self.simplemath = NODE_CLASS_MAPPINGS["SimpleMath+"]()
        self.imagepadkj = NODE_CLASS_MAPPINGS["ImagePadKJ"]()
        self.imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        self.nanobananabasicnode = NODE_CLASS_MAPPINGS["NanoBananaBasicNode"]()
        self.invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        self.imagecropbymask = NODE_CLASS_MAPPINGS["ImageCropByMask"]()
        self.dilateerodemask = NODE_CLASS_MAPPINGS["DilateErodeMask"]()
        self.easy_imagedetailtransfer = NODE_CLASS_MAPPINGS["easy imageDetailTransfer"]()
        self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        self.image_blend_by_mask = NODE_CLASS_MAPPINGS["Image Blend by Mask"]()

    def run(self, image, prompt, resolution='2K', detail=5.0, blend=0.2 ):

        # choose the processing resolution
        size = 1024 if resolution == "1K" else 2048 if resolution == "2K" else 4096

        with torch.inference_mode():
            loadimage_1 = self.loadimage.load_image(image=image)

            getimagesizeandcount_123 = self.getimagesizeandcount.getsize(
                image=get_value_at_index(loadimage_1, 0)
            )

            simplemath_125 = self.simplemath.execute(
                value="max(a,b)",
                a=get_value_at_index(getimagesizeandcount_123, 1),
                b=get_value_at_index(getimagesizeandcount_123, 2),
            )

            imagepadkj_2 = self.imagepadkj.pad(
                left=0,
                right=0,
                top=0,
                bottom=0,
                extra_padding=0,
                pad_mode="color",
                color="128, 128, 128",
                image=get_value_at_index(loadimage_1, 0),
                target_width=get_value_at_index(simplemath_125, 0),
                target_height=get_value_at_index(simplemath_125, 0),
            )

            imageresizekjv2_5 = self.imageresizekjv2.resize(
                width=size,
                height=size,
                upscale_method="bilinear",
                keep_proportion="resize",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=1,
                device="cpu",
                image=get_value_at_index(imagepadkj_2, 0),
                unique_id=9483597482083527186,
            )

            nanobananabasicnode_7 = self.nanobananabasicnode.run(
                gemini_key=os.getenv("GEMINI_KEY"),
                prompt=prompt,
                aspect_ratio="match_input_image",
                resolution=resolution,
                output_format="png",
                safety_filter_level="block_only_high",
                image_input=get_value_at_index(imageresizekjv2_5, 0),
            )

            imageresizekjv2_13 = self.imageresizekjv2.resize(
                width=get_value_at_index(simplemath_125, 0),
                height=get_value_at_index(simplemath_125, 0),
                upscale_method="lanczos",
                keep_proportion="resize",
                pad_color="0, 0, 0",
                crop_position="center",
                divisible_by=1,
                device="cpu",
                image=get_value_at_index(nanobananabasicnode_7, 0),
                unique_id=5704576468089321932,
            )

            invertmask_14 = self.invertmask.invert(mask=get_value_at_index(imagepadkj_2, 1))

            imagecropbymask_9 = self.imagecropbymask.crop(
                image=get_value_at_index(imageresizekjv2_13, 0),
                mask=get_value_at_index(invertmask_14, 0),
            )

            invertmask_130 = self.invertmask.invert(mask=get_value_at_index(loadimage_1, 1))

            dilateerodemask_24 = self.dilateerodemask.dilate_mask(
                radius=-5,
                shape="box",
                masks=get_value_at_index(invertmask_130, 0),
            )

            easy_imagedetailtransfer_104 = self.easy_imagedetailtransfer.transfer(
                mode="add",
                blur_sigma=detail,
                blend_factor=1,
                image_output="Preview",
                save_prefix="ComfyUI",
                target=get_value_at_index(imagecropbymask_9, 0),
                source=get_value_at_index(loadimage_1, 0),
                mask=get_value_at_index(dilateerodemask_24, 0),
            )

            masktoimage_114 = self.masktoimage.mask_to_image(
                mask=get_value_at_index(dilateerodemask_24, 0)
            )

            image_blend_by_mask_113 = self.image_blend_by_mask.image_blend_mask(
                blend_percentage=blend,
                image_a=get_value_at_index(easy_imagedetailtransfer_104, 0),
                image_b=get_value_at_index(loadimage_1, 0),
                mask=get_value_at_index(masktoimage_114, 0),
            )

            output_image = get_value_at_index(image_blend_by_mask_113, 0)
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
