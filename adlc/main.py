from adlc.adlc import ADLC
from fastapi import FastAPI, Depends, File, UploadFile
from pydantic import BaseModel
from typing import Annotated, Any
from PIL import Image

app = FastAPI()

# Although ADLC is not stateful, this will hopefully avoid
# instantiating large modules as often
@app.on_event("startup")
def start_adlc():
    app.state.adlc = ADLC()


async def get_adlc():
    return app.state.adlc


@app.post("/process_img/")
async def process_img(file:UploadFile = File(...), adlc=Annotated[ADLC, Depends(get_adlc)]) -> list[dict]:
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
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))   
    metadata = pil_image.getxmp()
    img_rgb = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)


    # Send image to ADLC wrapped in 1-elem list since KerasCV handles images in batches
    results = adlc.process_images([img_rgb], metadata)

    # Since only processing one image, unwrap
    return results[0]
