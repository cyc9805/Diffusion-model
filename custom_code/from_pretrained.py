# #%%
from diffusers import DDIMPipeline

model_id = "google/ddpm-celebahq-256"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
image = ddim(num_inference_steps=50).images

for i, im in enumerate(image):
    # save image
    im.save(f"/home/cyc/Diffusion_model/running_with_diffusers/results/ddim_generated_image{i}.png")
    if i > 5:
        break

# import torch
# from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  