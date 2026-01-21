# Standard library
import logging
import tempfile
import time
from io import BytesIO

# Standard typing
from typing import Annotated, Literal

# Third-party
from PIL import Image
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import Field, HttpUrl
import uvicorn

# Local application
from retouch_model import NanoBananaMaskedInpaint, NanoBananaBgChange
from retouch_utils import fetch_image_data, get_foreground_mask, get_rgba_image, mask_to_region


masked_inpainter = NanoBananaMaskedInpaint()
bg_changer = NanoBananaBgChange()

templates = Jinja2Templates(directory="templates")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(process)d - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
app = FastAPI(
    title="Retouch API",
    version="1.0.0",
    openapi_url="/retouch/openapi.json",
    docs_url="/retouch/docs",
    redoc_url="/retouch/redoc",
)

@app.get('/retouch/status')
def retouch_preset_status_get(request: Request):
    return {'status': 'OK'}


@app.get("/retouch/masked/generate")
def retouch_preset_predict_get(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


@app.post("/retouch/masked/generate")
async def retouch_preset_predict_post(request: Request, image: UploadFile = File(...), mask: UploadFile = File(None),
                                      image_url: HttpUrl = Form(None), mask_url: HttpUrl = Form(None),
                                      prompt: str = Form(...), resolution: Literal["1K", "2K", "4K"] = Form("1K"),
                                      square_region: bool = Form(True)):
    try:
        t1 = time.perf_counter()
        logger.info(f"prompt: {prompt}")

        image = fetch_image_data(image_url) if image_url  else await image.read()
        mask =  fetch_image_data(mask_url) if mask_url else await mask.read()

        if square_region:
            mask = mask_to_region(mask)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image, \
                tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_mask:

            temp_image.write(image)
            temp_image.flush()

            temp_mask.write(mask)
            temp_mask.flush()

            output = masked_inpainter.run(temp_image.name, temp_mask.name, prompt,resolution=resolution)

        t2 = time.perf_counter() - t1
        logger.info(f"time taken: {t2}")

        return Response(content=output, media_type="image/jpg",
                        headers={'Content-Disposition': f'attachment; filename=processed.jpg'})

    except Exception as e:
        logger.exception(str(e), exc_info=True)
        return {"code": -1, "status": f"prediction failed - {str(e)}"}

@app.get("/retouch/bgchange/generate")
def retouch_preset_predict_get(request: Request):
    return templates.TemplateResponse("upload_image.html", {"request": request})


@app.post("/retouch/bgchange/generate")
async def retouch_preset_predict_post(request: Request, image: UploadFile = File(...), mask: UploadFile = File(None),
                                      image_url: HttpUrl = Form(None), mask_url: HttpUrl = Form(None),
                                      prompt: str = Form(...),  resolution: Literal["1K", "2K", "4K"] = Form("2K"),
                                      detail: Annotated[float, Form(), Field(ge=0.1, le=10.0)] = 5.0,
                                      blend: Annotated[float, Form(), Field(ge=0.0, le=1.0)] = 0.2,
                                      ):
    try:
        t1 = time.perf_counter()
        logger.info(f"prompt: {prompt}")

        # expects rgb image
        image = fetch_image_data(image_url) if image_url  else await image.read()

        # if separate mask is provided, use the new alpha mask
        if  mask or mask_url:
            mask =  fetch_image_data(mask_url) if mask_url else await mask.read()
        else:
            mask = get_foreground_mask(image, mask_only=True)
        image = get_rgba_image(image, mask)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_image:

            temp_image.write(image)
            temp_image.flush()

            output = bg_changer.run(temp_image.name, prompt=prompt, resolution=resolution,
                                    detail=detail, blend=blend)

        t2 = time.perf_counter() - t1
        logger.info(f"time taken: {t2}")

        return Response(content=output, media_type="image/jpg",
                        headers={'Content-Disposition': f'attachment; filename=processed.jpg'})

    except Exception as e:
        logger.exception(str(e), exc_info=True)
        return {"code": -1, "status": f"prediction failed - {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("retouch_api:app", host="0.0.0.0", port=5007, workers=1)
