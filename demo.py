import base64
import json
import logging
from io import BytesIO

import requests
import gradio as gr
from PIL import Image, ImageChops

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(process)d - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

NANOBANANAPRO_URL = "https://generative-api-132358776415.asia-south1.run.app/generative-service/gemini/nanobananapro/generate"
BACKGROUND_CHANGE_URL = "https://retouch-api-132358776415.asia-south1.run.app/retouch/bgchange/generate"
RETOUCH_MASKED_URL = "https://retouch-api-132358776415.asia-south1.run.app/retouch/masked/generate"
SAM3_GEOMETRIC_URL = "https://bgo350nuo5kacq-80.proxy.runpod.net/sam3/segmentation/mask/geometric"
SAM3_TEXT_URL = "https://bgo350nuo5kacq-80.proxy.runpod.net/sam3/segmentation/mask/text"


# =====================================================
# UTILS
# =====================================================

def base64_to_bytes(base64_string):
    return base64.b64decode(base64_string)

def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes))
    return img.convert("RGB")

def pil_to_bytes(image: Image.Image, format="WEBP") -> bytes:
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def overlay_mask(img, mask, color=(255,0,0), opacity=0.3):
    img = img.convert("RGBA")
    mask = mask.convert("L")

    alpha = ImageChops.multiply(
        mask,
        Image.new("L", mask.size, int(255 * opacity))
    )

    overlay = Image.new("RGBA", img.size, (*color, 255))
    overlay.putalpha(alpha)

    return Image.alpha_composite(img, overlay)

def invert_mask(image, mask):
    mask = ImageChops.invert(mask)
    overlay = overlay_mask(image, mask)
    return overlay, mask

# =====================================================
# BACKGROUND CHANGE
# =====================================================

def get_bgchange_prompt(bg_prompt: str) -> str:
    return f"""
E-commerce photo edit. Maintain the original subject (person or product) EXACTLY as it appears in the source image, including its exact pose, facial features, body proportions, framing, and camera depth.

The subject must remain 100% unchanged. Garment mask is locked — DO NOT modify garment pixels, colors, textures, patterns, or fit. Clothing stays 100% unchanged.

Replace the entire background with {bg_prompt}. The new background should be vibrant and naturally lit.

Blend lighting, shadows, and perspective seamlessly so the original subject looks naturally integrated into the new environment.

Photorealistic output, clean catalog/studio aesthetic, consistent lighting, zero distortions or cloning artifacts.

Note: Keep the gray padded area at the border exactly same, do not alter the gray padded area. Also, do not add extra gray padding or border at the borders.

CRITICAL: Prevent Scaling and 'Un-cropping'

Exact Size & Scale: Maintain the subject's exact size and pixel dimensions relative to the canvas from the input image. DO NOT scale down, shrink, or resize the subject to fit the new background.

Respect Original Frame Boundaries: The subject must remain cropped exactly as it is in the source image. If the subject's body or the object is cut off by any edge of the original frame (top, bottom, left, or right), DO NOT generate or invent the missing parts to complete the figure.

Negative Constraint: Do not add any body parts, hair, or product parts that are not currently visible in the source image pixels. The new background must fill the canvas around the existing subject crop; it must not expand the subject itself.

Goal: Background replacement only, with 100% preservation of the original subject, garments, and position.
"""

def background_change(image: Image.Image, prompt: str,
    detail: float = 5.0, blend: float = 0.2, resolution: str = "2K"):

    try:
        image_bytes = pil_to_bytes(image)
        bg_prompt = get_bgchange_prompt(prompt)

        files = {"image": image_bytes}
        data = { "prompt": bg_prompt, "resolution": resolution, "detail": detail, "blend": blend}

        response = requests.post(BACKGROUND_CHANGE_URL, files=files, data=data, timeout=360)
        response.raise_for_status()

        output = bytes_to_pil(response.content)
        logger.info(bg_prompt)
        logger.info(f"background change successfull")
        return image, output
    except Exception as e:
        logger.exception(str(e))
        gr.Warning("Something went wrong!")
        return image, image

# =====================================================
# NANOBANANA RETOUCH
# =====================================================

def generate_image(image: Image.Image, prompt: str):

    try:
        image_bytes = pil_to_bytes(image)

        files = {"images": image_bytes}
        data = {"prompt": prompt, "aspect_ratio": "custom", "resolution": "match_input", "format": "WEBP"}
        logger.info(data)

        response = requests.post(NANOBANANAPRO_URL,files=files,data=data,timeout=360)
        if not isinstance(response.content, bytes):
            logger.warning(f"invalid response: {response.content}")
        response.raise_for_status()
        output = bytes_to_pil(response.content)
        logger.info(f"text edit successfull")
        return image, output
    except Exception as e:
        logger.exception(str(e))
        gr.Warning("Something went wrong!")
        return image, image

def extract_image_and_mask(editor_output):
    background = editor_output["background"]
    layer = editor_output["layers"][-1]

    image = Image.fromarray(background.astype("uint8")).convert("RGB")
    mask = layer[..., 3]  # (H, W)

    mask = Image.fromarray((mask).astype("uint8")).convert("L")
    return image, mask


def generate_masked(masked_image: dict, prompt: str):

    image, mask = extract_image_and_mask(masked_image)

    try:
        image_bytes = pil_to_bytes(image)
        mask_bytes = pil_to_bytes(mask, format="PNG")

        files = { "image": image_bytes, "mask":mask_bytes}
        data = {"prompt": prompt}

        response = requests.post(RETOUCH_MASKED_URL, files=files, data=data, timeout=360)
        if not isinstance(response.content, bytes):
            logger.warning(f"invalid response: {response.content}")
        response.raise_for_status()

        output = bytes_to_pil(response.content)
        logger.info(f"masked edit successfull")
        return image, output
    except Exception as e:
        logger.exception(str(e))
        gr.Warning("Something went wrong!")
        return image, image

# =====================================================
# SAM3 RETOUCH
# =====================================================

def generate_masked_sam(image, mask, prompt: str):

    try:
        image_bytes = pil_to_bytes(image)
        mask_bytes = pil_to_bytes(mask, format="PNG")

        files = { "image": image_bytes, "mask":mask_bytes}
        data = {"prompt": prompt}

        response = requests.post( RETOUCH_MASKED_URL, files=files, data=data, timeout=360)
        if not isinstance(response.content, bytes):
            logger.warning(f"invalid response: {response.content}")
        response.raise_for_status()

        output = bytes_to_pil(response.content)
        logger.info(f"masked edit successfull")
        return image, output
    except Exception as e:
        logger.exception({str(e)})
        gr.Warning("Something went wrong!")
        return image, image


def process_clicks(image: Image.Image, sam_prompt_state, click_type: str,evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    label = 1 if click_type == "positive" else 0
    width, height = image.size

    sam_prompt_state['points'].append([x/width,y/height])
    sam_prompt_state['labels'].append(label)

    mask = get_sam3_geometric_mask(image,prompt=sam_prompt_state)
    overlay = overlay_mask(image, mask)
    return overlay, sam_prompt_state ,mask

def process_prompts(image: Image.Image, text_prompt: str):
    mask = get_sam3_text_mask(image,prompt=text_prompt)
    overlay = overlay_mask(image, mask)
    return overlay, mask


def get_sam3_geometric_mask(image: Image.Image, prompt: dict) -> Image.Image:

    try:
        prompt = json.dumps(prompt)
        image_bytes = pil_to_bytes(image)

        files = {"image": image_bytes}
        data = {"prompt": prompt}
        logger.info(data)

        response = requests.post(SAM3_GEOMETRIC_URL, files=files, data=data, timeout=360)
        response.raise_for_status()

        # extract mask and state
        output = response.json()
        mask_base64 = output['curr_mask']
        mask_bytes = base64_to_bytes(mask_base64)

        output = bytes_to_pil(mask_bytes)
        logger.info(f"click mask: masked generation successfull")
        return output
    except Exception as e:
        logger.exception(str(e))
        gr.Warning("Something went wrong!")
        return image

def get_sam3_text_mask(image: Image.Image, prompt: str) -> Image.Image:

    try:
        image_bytes = pil_to_bytes(image)

        files = {"image": image_bytes}
        data = {"prompt": prompt}
        logger.info(data)

        response = requests.post(SAM3_TEXT_URL, files=files, data=data, timeout=360)
        response.raise_for_status()

        # extract mask and state
        output = response.json()
        assert 'output' in output, f"invalid mask: {output}"

        mask_base64 = output['output'][0]['mask']
        mask_bytes = base64_to_bytes(mask_base64)

        output = bytes_to_pil(mask_bytes)
        logger.info(f"text mask: mask generation successfull")
        return output
    except Exception as e:
        logger.exception(str(e))
        gr.Warning("Something went wrong!")
        return image

# =====================================================
# GRADIO UI
# =====================================================

def build_ui():
    with gr.Blocks(title="Photogenix Edit") as demo:
        gr.Markdown(
            "<h2 style='text-align:center;'>Photogenix Retouch</h2>"
        )

        with gr.Tabs():
            with gr.Tab("Text Edit"):
                # Row 1: input image | output image
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=500
                    )
                    output_slider = gr.ImageSlider(
                        label="Before / After",
                        type="pil",
                        height=500
                    )

                # Row 2: prompt | generate button (below output)
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the edit..."
                    )

                    with gr.Column():
                        gr.Markdown("")  # spacer
                        generate_btn = gr.Button("Generate")

                generate_btn.click(
                    fn=generate_image,
                    inputs=[input_image, prompt],
                    outputs=output_slider
                )

            # ---------- ADVANCED TAB ----------
            with gr.Tab("Mask Edit"):
                with gr.Row():
                    masked_input = gr.ImageEditor(
                        label="Edit Image",
                        brush=gr.Brush(
                            colors=["#ffffff"],  # single color
                        ),
                        interactive=True,
                        height=500,
                    )

                    advanced_output = gr.ImageSlider(
                        label="Before / After",
                        type="pil",
                        height=500
                    )

                with gr.Row():
                    advanced_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the masked edit..."
                    )

                    with gr.Column():
                        gr.Markdown("")
                        generate_masked_btn = gr.Button("Generate")

                generate_masked_btn.click(
                    fn=generate_masked,
                    inputs=[masked_input, advanced_prompt],
                    outputs=advanced_output
                )

            # ---------- BG CHANGE TAB ----------
            with gr.Tab("BG Change"):
                with gr.Row():
                    bg_input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=500
                    )
                    bg_output_slider = gr.ImageSlider(
                        label="Before / After",
                        type="pil",
                        height=500
                    )

                with gr.Row():
                    bg_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe new background..."
                    )

                # Detail + Blend sliders
                with gr.Row():
                    detail_slider = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=0.1,
                        label="Detail"
                    )
                    blend_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.2,
                        step=0.01,
                        label="Blend"
                    )

                with gr.Row():
                    bg_generate_btn = gr.Button("Generate")

                bg_generate_btn.click(
                    fn=background_change,  # your backend caller
                    inputs=[bg_input_image, bg_prompt, detail_slider, blend_slider],
                    outputs=bg_output_slider
                )

            # SAM3 Segmentation
            with gr.Tab("Smart Edit"):
                # =========================
                # ROW 1 — Images
                # =========================
                with gr.Row():
                    with gr.Column(scale=1):
                        sam_input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=500
                        )

                    with gr.Column(scale=1):
                        output_slider = gr.ImageSlider(
                            label="Before / After",
                            type="pil",
                            height=500
                        )

                # =========================
                # ROW 2 — Prompt + Generate
                # =========================
                with gr.Row():
                    with gr.Group():
                        # left column (under image)
                        with gr.Column(scale=1):
                            mask_prompt = gr.Textbox(
                                label="Mask Prompt",
                                placeholder="Describe the required object or region"
                            )
                            with gr.Row(scale=1):
                                sam_text_generate_btn = gr.Button("Generate Mask")

                    # right column (under slider)
                    with gr.Column(scale=1):
                        with gr.Group():
                            # left column (under image)
                            with gr.Column(scale=1):
                                edit_prompt = gr.Textbox(
                                    label="Edit Prompt",
                                    placeholder="Describe the edit required for masked region"
                                )
                                sam_image_generate_btn = gr.Button("Generate Image")

                # =========================
                # ROW 3 — Click controls
                # =========================
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            with gr.Column(scale=1):
                                click_type = gr.Radio(
                                    ["positive", "negative"],
                                    value="positive",
                                    label="Click Type",
                                )
                            with gr.Column(scale=1):
                                reset_btn = gr.Button("Reset Mask", size="md")
                                invert_btn = gr.Button("Invert Mask", size="md")

                    with gr.Column(scale=1):
                        gr.Markdown("")

                # =========================
                # Events and States
                # =========================

                # init prompt state
                sam_prompt_state = gr.State({
                    "points": [],
                    "labels": []
                })

                # init image state
                sam_orig_image_state = gr.State()
                sam_mask_state = gr.State()

                # reset states on image upload
                sam_input_image.upload(
                    lambda img: (img, {"points": [], "labels": []}),
                    sam_input_image,
                    [sam_orig_image_state, sam_prompt_state]
                )

                # reset clicks and overlay
                reset_btn.click(
                    lambda img: (img, {"points": [], "labels": []}),
                    inputs=sam_orig_image_state,
                    outputs=[sam_input_image, sam_prompt_state],
                    show_progress="hidden"
                )

                # invert current mask selection
                invert_btn.click(
                    fn=invert_mask,
                    inputs=[sam_orig_image_state, sam_mask_state],
                    outputs=[sam_input_image, sam_mask_state],
                    show_progress="hidden"
                )

                # image generation based on mask
                sam_image_generate_btn.click(
                    fn=generate_masked_sam,
                    inputs=[sam_orig_image_state, sam_mask_state, edit_prompt],
                    outputs=output_slider
                )

                # mask generation based on text prompt
                sam_text_generate_btn.click(
                    fn=process_prompts,
                    inputs=[sam_orig_image_state, mask_prompt],
                    outputs=[sam_input_image, sam_mask_state]
                )

                # always pass original input image for sam3 processing, overwrite upload image ui widget with final overlay image
                sam_input_image.select(process_clicks, [sam_orig_image_state, sam_prompt_state, click_type],
                                       [sam_input_image, sam_prompt_state, sam_mask_state])
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True, root_path='/gradio')

