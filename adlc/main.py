from adlc.adlc import ADLC
from fastapi import FastAPI, Depends, File, UploadFile
from pydantic import BaseModel
from typing import Annotated, Any
from PIL import Image
from io import BytesIO
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

import cv2
import numpy as np


# Although ADLC is not stateful, this will hopefully avoid
# instantiating large modules as often
@asynccontextmanager
async def lifespan(app: FastAPI):
    adlc_instance = ADLC()
    yield {'adlc_instance': adlc_instance}

app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.post("/process_img/")
async def process_img(request: Request, file:UploadFile = File(...)) -> list[dict]:
    """
    Given a PIL image object, returns a list with a dict for each target found in the image

    Currently each dict is in the form::

        {
            "xywh": (x,y,w,h),
            "bg_color": bg_color,
            "txt_color": txt_color,
            "predicted_character": char,
            "predicted_shape": shape,
            "longitude": long,
            "latitude": lat,
        }
    """
    # Read in image from body as PIL
    contents = await file.read()
    pil_image = Image.open(BytesIO(contents))   
    
    # Unwrap XMP data
    metadata = pil_image.getxmp()["xmpmeta"]["RDF"]["Description"]
    
    # Convert image to opencv (ndarray)
    img_rgb = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)

    # Send image to ADLC wrapped in 1-elem list since KerasCV handles images in batches
    adlc_instance = request.state.adlc_instance
    results = adlc_instance.process_images([img_rgb], metadata)

    return results
