import sys

import cv2
import numpy as np
import torch
from compel import Compel  # type: ignore
from diffusers.pipelines.auto_pipeline import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "mps"

# Elon Muskに顔のみを変換するためのベストなStable Difusionのプロンプトを記述
prompt = "a photo of elon musk"

negative_prompt = ""

pipe_t2i: StableDiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float, variant="fp16"
).to(device)

pipe: StableDiffusionImg2ImgPipeline = AutoPipelineForImage2Image.from_pipe(pipe_t2i)

if not isinstance(pipe, StableDiffusionImg2ImgPipeline):
    sys.exit(1)

cap = cv2.VideoCapture(0)

img_dst = Image.new("RGB", (1024, 512))

generator = torch.Generator(device).manual_seed(42)
compel_proc = Compel(tokenizer=pipe_t2i.tokenizer, text_encoder=pipe_t2i.text_encoder)
prompt_embeds = compel_proc(prompt)
negative_prompt_embeds = compel_proc(negative_prompt)

while True:
    ret, frame = cap.read()

    width, height = frame.shape[1], frame.shape[0]
    left = (width - 1024) // 2
    top = (height - 1024) // 2
    right = (width + 1024) // 2
    bottom = (height + 1024) // 2

    img_init = Image.fromarray(frame).crop((left, top, right, bottom)).resize((512, 512), Image.NEAREST)

    result = pipe(
        image=img_init,
        num_inference_steps=2,
        guidance_scale=0,
        generator=generator,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    if isinstance(result, StableDiffusionPipelineOutput):
        img_result = result.images[0]

        img_dst.paste(img_init, (0, 0))
        img_dst.paste(img_result, (512, 0))

        cv2.imshow("img_dst", np.array(img_dst))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
