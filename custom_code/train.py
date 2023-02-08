import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils import *
from modules import *
from tqdm import tqdm

def train(
    dataloader, 
    model, 
    eval_batch_size, 
    noise_scheduler, 
    num_epochs, 
    lr, 
    lr_warmup_steps, 
    num_inference_steps,
    cur_dir, 
    device): 

    model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=lr_warmup_steps,
        # len(dataloader) is equal to # of batches
        num_training_steps=(len(dataloader) * num_epochs)
        )
    
    for epoch in range(1, num_epochs+1):
        print(f'========== starting epoch {epoch}/{num_epochs} ============')
        for i, (image, label) in tqdm(enumerate(dataloader)):

            image = image.to(device)
            num_batch = image.shape[0]

            # Sample random gaussian noise
            noise = torch.randn(image.shape).to(device)

            # Sample random timestep
            timestep = torch.randint(0, noise_scheduler.num_train_timesteps, (num_batch,)).to(device)

            # Add noise to the image
            noisy_image = noise_scheduler.add_noise(image, noise, timestep)

            # Predict the noise added to the image
            pred_noise = model(sample=noisy_image, timestep=timestep, return_dict=False)[0]

            # Calculate loss and backpropagate
            optimizer.zero_grad()
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()

            # Scehduling
            lr_scheduler.step()


        # infer when epoch reaches multiple of 5
        if epoch == 1 or epoch % 5 == 0:
            torch.save(model, os.path.join(cur_dir, "models", f"epoch{epoch}-ckpt.pt"))
            pipeline = DDPMPipeline(unet=model.module, scheduler=noise_scheduler)
            evaluate(pipeline=pipeline, epoch=epoch, eval_batch_size=eval_batch_size, num_inference_steps=num_inference_steps, cur_dir=cur_dir)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=128, help='size of the image')
    parser.add_argument('--train_batch_size', default=10, help='train batch size')
    parser.add_argument('--eval_batch_size', default=3, help='eval batch size')
    parser.add_argument('--num_epochs', default=50, help='number of epoch')
    parser.add_argument('--learning_rate', default=1e-4, help='learning rate')
    parser.add_argument('--lr_warmup_steps', default=500, help='lr_warmup_steps')
    parser.add_argument('--num_inference_steps', default=1000, help='number of denoising steps')
    parser.add_argument('--dataset_name', default='CelebA', help='name of the dataset')

    opt = parser.parse_args()

    image_size = opt.image_size
    train_batch_size = opt.train_batch_size
    eval_batch_size = opt.eval_batch_size
    num_epochs = opt.num_epochs
    lr = opt.learning_rate
    lr_warmup_steps = opt.lr_warmup_steps
    num_inference_steps = opt.num_inference_steps
    dataset_name = opt.dataset_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unconditional_model = create_model(input_image_size=image_size, input_channel_size=3)
    train_dataloader = load_data(image_size=image_size, batch_size=train_batch_size, dataset_name=dataset_name)
    noise_scheduler = scheduler()
    cur_dir = '/home/cyc/Diffusion_model/running_with_diffusers'


    train(
        train_dataloader, 
        unconditional_model, 
        eval_batch_size, 
        noise_scheduler, 
        num_epochs, lr, 
        lr_warmup_steps,
        num_inference_steps,
        cur_dir,
        device)

    
if __name__ == '__main__':
    exit(main())




