from typing import Literal, Union

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

ModelName = Literal["stabilityai/sd-turbo", "stabilityai/sdxl-turbo"]


class SdTurbo:
    def __init__(self, model_name: ModelName):
        device = "cuda" if torch.cuda.is_available() else "mps"

        pipe_t2i: StableDiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
            model_name, torch_dtype=torch.float, variant="fp16"
        ).to(device)

        pipe: StableDiffusionImg2ImgPipeline = AutoPipelineForImage2Image.from_pipe(pipe_t2i)

        if not isinstance(pipe, StableDiffusionImg2ImgPipeline):
            raise Exception("pipe is not StableDiffusionImg2ImgPipeline")

        self.pipe = pipe

        self.generator = torch.Generator(device).manual_seed(42)

        self.compel_proc = Compel(tokenizer=pipe_t2i.tokenizer, text_encoder=pipe_t2i.text_encoder)

    def prompt_embeds(self, prompt: str):
        return self.compel_proc(prompt)

    def run(self, prompt: str, negative_prompt: str, image: Union[np.ndarray, Image.Image]):
        prompt_embeds = self.prompt_embeds(prompt)
        negative_prompt_embeds = self.prompt_embeds(negative_prompt)

        result = self.pipe(
            image=image,
            num_inference_steps=2,
            guidance_scale=0,
            generator=self.generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if isinstance(result, StableDiffusionPipelineOutput):
            return result.images[0]
