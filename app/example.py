# https://huggingface.co/docs/diffusers/optimization/mps#metal-performance-shaders-mps
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
# pipe.enable_attention_slicing()

prompt = "a photo of tyler the creator"
image = pipe(prompt).images[0]

image.save("out.png")
