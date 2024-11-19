# -*- coding: utf-8 -*-
import numpy as np
import argparse
from copy import deepcopy
import sys
import platform

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from omegaconf import OmegaConf

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    sys.path.append('/home/lars/Outliers/nonlinear-outlier-synthesis/')
else:
    sys.path.append('/storage/homefs/ld20d359/nonlinear-outlier-synthesis/')

from file_size_estimation.fse.trainer import load_model_from_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


anchors = torch.from_numpy(np.load('./token_embed_c10.npy')).cuda()
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
prompts = ['An image of a ' + class_name for class_name in class_names]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--num_images", type=int, default=10000)
args = parser.parse_args()

print(f"Offset: {args.offset}, Num images: {args.num_images}")

config = OmegaConf.load(f"configs/stable-diffusion/v1-inference.yaml")
config["model"]["params"]["unet_config"]["params"]["use_checkpoint"] = False

model = load_model_from_config(config, f"/storage/workspaces/artorg_aimi/ws_00000/lars/sd-v1-4.ckpt")
model = model.to(device)
model.eval()

# Wrap in DataParallel if more than 1 GPU is available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

original_embed = deepcopy(model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight)

# Set requires grad to False for all parameters in model.model and model.first_stage_model
for param in model.model.parameters():
    param.requires_grad = False

for param in model.first_stage_model.parameters():
    param.requires_grad = False

# Set requires grad to False for all parameters in model.cond_stage_model except for the embedding
for name, param in model.cond_stage_model.named_parameters():
    if name != 'token_embedding':
        param.requires_grad = False
    else:
        param.requires_grad = True

# Print number of parameters with requires_grad=True
print(f"Number of parameters with requires_grad=True: {sum(p.numel() for p in model.cond_stage_model.parameters() if p.requires_grad)}")

train_data_in = torchvision.datasets.CIFAR10("../data/", train=True, transform=transform)

batch_size = 32

image_embeddings = []
labels = []
# Loop over training data
for cnt, (image, label) in enumerate(train_data_in):
    if cnt < args.offset:
        continue

    if cnt >= args.offset + args.num_images:
        break

    # # Plot image
    # import matplotlib.pyplot as plt
    # plt.imshow((image.permute(1, 2, 0) + 1) / 2)
    # plt.savefig(f"image_{cnt}.png")

    # Find class token
    tmp_token = model.cond_stage_model.tokenizer([class_names[label]], truncation=True, max_length=model.cond_stage_model.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = tmp_token["input_ids"].to(device)
    original_id = tokens[0][1]

    # Re-initialize the trained embedding layer and optimizer. 
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight = nn.Parameter(original_embed.clone())
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.requires_grad = True

    optimizer = torch.optim.Adam(model.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters(), lr=1e-2)

    losses = []

    with torch.no_grad():
        # Transform image
        image = image.unsqueeze(0).to(device)
        
        # Encode image to latent space
        z = model.encode_first_stage(image).mode() * 0.18215

        # Duplicate the image to batch_size
        z = z.repeat(batch_size, 1, 1, 1)

    for epoch in range(3):
        # Embed the condition
        cc = model.cond_stage_model.forward_imgembs(prompts[label]).to(device).repeat(batch_size, 1, 1)

        # Noise latent space
        t = torch.randint(0, 50, size=(batch_size,), device=device) * 20 + 1
        noise = torch.randn_like(z)

        cumalphas_t = model.alphas_cumprod[t][:, None, None, None]
        xt = torch.sqrt(cumalphas_t)*z + torch.sqrt(1 - cumalphas_t)*noise

        # Predict noise
        e_t = model.model.diffusion_model(xt, timesteps=t, context=cc)

        # Compute MSE loss with optional L2 regularization
        optimizer.zero_grad()
        loss = ((e_t - noise)**2).mean() + 0.0 * (model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight[original_id] - anchors[label]).pow(2).sum()
        loss.backward()

        # Set the gradients for all tokens except the class token to zero
        model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.grad[torch.arange(model.cond_stage_model.transformer.\
                                                                                                          text_model.embeddings.token_embedding.weight.grad.shape[0], device=device) != original_id] = 0

        optimizer.step()
        losses.append(loss.item())

    image_embedding = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight[original_id].detach().cpu()
    image_embeddings.append(image_embedding)

    labels.append(label)

    if cnt % 25 == 0:
        print(f"Last loss: {losses[-1]}. Processed {cnt} images", flush=True)
                        
image_embeds = torch.stack(image_embeddings).numpy()
print(image_embeds.shape)

labels = np.array(labels)

np.save \
        (f"embed_c10_images_clip_{args.offset}_{args.num_images}.npy", image_embeds)
