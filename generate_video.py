
import os
from omegaconf import OmegaConf
import torch
# from torch import autocast
from torch.cuda.amp import autocast
from diffusers import DDIMScheduler
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
from natsort import natsorted
from glob import glob

#@markdown Name/Path of the initial model.
# MODEL_NAME = "CompVis/stable-diffusion-v1-4" #@param {type:"string"}

#@markdown If model should be download from a remote repo. Untick it if the model is loaded from a local path.
def download_pretrained_model(MODEL_NAME):
    #@markdown If model should be download from a remote repo. Untick it if the model is loaded from a local path.
    download_pretrained_model = True #@param {type:"boolean"}
    if download_pretrained_model:
        os.system(f'git clone https://huggingface.co/{MODEL_NAME} checkpoints/{MODEL_NAME}')
        MODEL_NAME = f"./checkpoints/{MODEL_NAME}"
    print(f"[*] MODEL_NAME={MODEL_NAME}")

def save_to_dir(OUTPUT_DIR):
    print("hihihihi")
    os.system(f'mkdir -p {OUTPUT_DIR}')
    print(f"[*] Weights will be saved at {OUTPUT_DIR}")

def train_config(THEME, TRAIN_PROMPT , OUTPUT_DIR, MODEL_NAME):
    CONFIG_NAME = f"configs/{THEME}.yaml" #@param {type:"string"}
    train_video_path = f"data/{THEME}.mp4" #@param {type:"string"}
    train_prompt = TRAIN_PROMPT #@param {type:"string"}
    video_length = 8 #@param {type:"number"}
    width = 512 #@param {type:"number"}
    height = 512 #@param {type:"number"}
    learning_rate = 3e-5 #@param {type:"number"}
    train_steps = 300 #@param {type:"number"}

    config = {
    "pretrained_model_path": MODEL_NAME,
    "output_dir": OUTPUT_DIR,
    "train_data": {
        "video_path": train_video_path,
        "prompt": train_prompt,
        "n_sample_frames": video_length,
        "width": width,
        "height": height,
        "sample_start_idx": 0,
        "sample_frame_rate": 2,
    },
    "validation_data": {
        "prompts": [
        "mickey mouse is skiing on the snow",
        "spider man is skiing on the beach, cartoon style",
        "wonder woman, wearing a cowboy hat, is skiing",
        "a man, wearing pink clothes, is skiing at sunset",
        ],
        "video_length": video_length,
        "width": width,
        "height": height,
        "num_inference_steps": 20,
        "guidance_scale": 12.5,
        "use_inv_latent": True,
        "num_inv_steps": 50,
    },
    "learning_rate": learning_rate,
    "train_batch_size": 1,
    "max_train_steps": train_steps,
    "checkpointing_steps": 1000,
    "validation_steps": 100,
    "trainable_modules": [
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ],
    "seed": 33,
    "mixed_precision": "fp16",
    "use_8bit_adam": False,
    "gradient_checkpointing": True,
    "enable_xformers_memory_efficient_attention": True,
    }
    OmegaConf.save(config, CONFIG_NAME)
    #VERYVERY SLOW
    os.system(f'accelerate launch train_tuneavideo.py --config={CONFIG_NAME}')
    print("FINALLY_FINISHED")


def get_result(OUTPUT_DIR, MODEL_NAME):
     # unet = UNet3DConditionModel.from_pretrained(OUTPUT_DIR, subfolder='unet', torch_dtype=torch.float16).to('cuda') #TODO
    # print("/Users/yejinlee/Desktop/4-1/AI/Tune-A-Video/checkpoints/CompVis/stable-diffusion-v1-4/unet/diffusion_pytorch_model.bin")
    # OUTPUT_DIR = "checkpoints/CompVis/stable-diffusion-v1-4/vae/"
    unet = UNet3DConditionModel.from_pretrained(OUTPUT_DIR, subfolder='unet', torch_dtype=torch.float16).to('cuda')
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder='scheduler')
    pipe = TuneAVideoPipeline.from_pretrained(MODEL_NAME, unet=unet, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    g_cuda = torch.Generator(device='cuda') #TODO
    seed = 1234 #@param {type:"number"}
    g_cuda.manual_seed(seed)

    prompt = "iron man is skiing" #@param {type:"string"}
    negative_prompt = "" #@param {type:"string"}
    use_inv_latent = True #@param {type:"boolean"}
    inv_latent_path = "" #@param {type:"string"}
    num_samples = 1 #@param {type:"number"}
    guidance_scale = 12.5 #@param {type:"number"}
    num_inference_steps = 50 #@param {type:"number"}
    video_length = 8 #@param {type:"number"}
    height = 512 #@param {type:"number"}
    width = 512 #@param {type:"number"}

    ddim_inv_latent = None
    if use_inv_latent and inv_latent_path == "":
        inv_latent_path = natsorted(glob(f"{OUTPUT_DIR}/inv_latents/*"))[-1]
        ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
        print(f"DDIM inversion latent loaded from {inv_latent_path}")

    with autocast("cuda"), torch.inference_mode():
        videos = pipe(
            prompt, 
            latents=ddim_inv_latent,
            video_length=video_length, 
            height=height, 
            width=width, 
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_samples,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).videos

    save_dir = "./results" #@param {type:"string"}
    save_path = f"{save_dir}/{prompt}.gif"
    save_videos_grid(videos, save_path)

    # display
    from IPython.display import Image, display
    display(Image(filename=save_path))

def main():
    print("hi")
    MODEL_NAME = "CompVis/stable-diffusion-v1-4"
    THEME= "man-skiing"
    TRAIN_PROMPT = "a man is skiing"
    OUTPUT_DIR = f"outputs/{THEME}" 
    download_pretrained_model(MODEL_NAME)
    save_to_dir(OUTPUT_DIR)
    train_config(THEME, TRAIN_PROMPT, OUTPUT_DIR, MODEL_NAME)
    get_result(OUTPUT_DIR, MODEL_NAME)

if __name__ == "__main__":
    main()
    exit()
