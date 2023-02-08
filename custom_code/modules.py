from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers import DDPMScheduler
from torch import load
import os

def create_model(input_image_size, input_channel_size, conditional=False):
    
    # refer to https://huggingface.co/docs/diffusers/api/models#diffusers.UNet2DModel
    # conditional
    if conditional:
        model = UNet2DConditionModel()
    
    # unconditional
    else:
        # model = UNet2DModel(
        #     sample_size = input_image_size,
        #     in_channels = channel_size,
        #     out_channels = channel_size,
        #     layers_per_block = 2,
        #     block_out_channels=(64, 128, 256, 256, 512, 512),
        #     time_embedding_type='positional',
        #     down_block_types=(
        #         'DownBlock2D',
        #         'AttnDownBlock2D',
        #         'DownBlock2D',
        #         'AttnDownBlock2D',
        #         'DownBlock2D',
        #         'AttnDownBlock2D',
        #     ),
        #     up_block_types=(
        #         'UpBlock2D',
        #         'AttnUpBlock2D',
        #         'UpBlock2D',
        #         'AttnUpBlock2D',
        #         'UpBlock2D',
        #         'AttnUpBlock2D'
        #     )
        # )
    
        model = UNet2DModel(
    sample_size=input_image_size,  # the target image resolution
    in_channels=input_channel_size,  # the number of input channels, 3 for RGB images
    out_channels=input_channel_size,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    block_out_channels=(256, 256, 512, 512, 1024, 1024),  # the number of output channels for each UNet block
    time_embedding_type='positional',
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      )
        )
    return model


def load_model(cur_dir, model_name):
    model_path = os.path.join(cur_dir, 'models', model_name)
    # model_path = os.path.join(cur_dir, model_name)
    model = load(model_path)
    return model
    
def scheduler():
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    return noise_scheduler


def evaluate(pipeline, epoch, eval_batch_size, num_inference_steps, cur_dir):
    images = pipeline(batch_size=eval_batch_size, num_inference_steps=num_inference_steps, return_dict=False)
    for i, image in enumerate(images[0]):
        save_dir = os.path.join(cur_dir, 'results', f'{i}th_{epoch}epoch.png')
        image.save(save_dir, 'PNG')
    print(f'successfully created {eval_batch_size} images!')

def show_image(cur_dir, image_name):
    pass
    