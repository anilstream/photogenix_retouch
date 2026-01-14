# Standard library
import logging

from io import BytesIO

# Third-party
import gradio as gr
import requests
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(process)d - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)



def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes))
    return img.convert("RGB")

def pil_to_bytes(image: Image.Image, format="PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


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
    

def generate_image(image: Image.Image, prompt: str,
    url: str = "https://generative-api-132358776415.asia-south1.run.app/generative-service/gemini/nanobananapro/generate"):

    try:
        input = pil_to_bytes(image)

        files = {"images": input}
        data = {"prompt": prompt, "aspect_ratio": "custom", "resolution": "match_input", "format": "WEBP"}

        response = requests.post(url,files=files,data=data,timeout=360)
        response.raise_for_status()
        print(response.content)
        output = bytes_to_pil(response.content)
        logger.info(f"text edit successfull")
    except Exception as e:
        logger.exception(str(e))
    return image, output

def background_change(image: Image.Image, prompt: str,
    detail: float = 5.0, blend: float = 0.2, resolution: str = "2K",
    url: str = "https://retouch-api-132358776415.asia-south1.run.app/retouch/bgchange/generate"):

    try:
        input = pil_to_bytes(image)
        prompt = get_bgchange_prompt(prompt)

        files = {"image": input}
        data = { "prompt": prompt, "resolution": resolution, "detail": detail, "blend": blend}

        response = requests.post(url, files=files, data=data, timeout=360)
        response.raise_for_status()

        output = bytes_to_pil(response.content)
        logger.info(bg_prompt)
        logger.info(f"background change successfull")
    except Exception as e:
        logger.exception(str(e))
    return image, output

def extract_image_and_mask(editor_output):
    background = editor_output["background"]
    layer = editor_output["layers"][-1]

    image = Image.fromarray(background.astype("uint8")).convert("RGB")
    mask = layer[..., 3]  # (H, W)

    mask = Image.fromarray((mask).astype("uint8")).convert("L")
    return image, mask


def generate_masked(masked_image: dict,prompt: str,
    url: str = "https://retouch-api-132358776415.asia-south1.run.app/retouch/masked/generate"):
    try:
        image, mask = extract_image_and_mask(masked_image)
        image_bytes = pil_to_bytes(image)
        mask_bytes = pil_to_bytes(mask)

        files = { "image": image_bytes, "mask":mask_bytes}
        data = {"prompt": prompt}

        response = requests.post( url, files=files, data=data, timeout=360)
        response.raise_for_status()

        output = bytes_to_pil(response.content)
        logger.info(f"masked edit successfull")
    except Exception as e:
        logger.exception(str(e))
    return image, output


# Gradio App
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

if __name__ == "__main__":
    demo.launch(share=True)

