# -*- coding: utf-8 -*-
import numpy as np
import argparse
from copy import deepcopy
import sys
import platform
import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from omegaconf import OmegaConf

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    sys.path.append('/home/lars/Outliers/nonlinear-outlier-synthesis/')
else:
    sys.path.append('/storage/homefs/ld20d359/nonlinear-outlier-synthesis/')

from file_size_estimation.fse.trainer import load_model_from_config

def get_model_attr(model, attr_name):
    if isinstance(model, torch.nn.DataParallel):
        return getattr(model.module, attr_name)
    else:
        return getattr(model, attr_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


anchors = torch.from_numpy(np.load('./token_embed_drd.npy')).cuda()
class_names = ["healthy retina", "mild diabetic retinopathy", "moderate diabetic retinopathy"]
prompts = ['An ' + class_name for class_name in class_names]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--num_images", type=int, default=10000)
parser.add_argument("--iterations", type=int, default=3)
args = parser.parse_args()

print(f"Offset: {args.offset}, Num images: {args.num_images}")

config = OmegaConf.load(f"configs/stable-diffusion/v1-inference.yaml")
config["model"]["params"]["unet_config"]["params"]["use_checkpoint"] = False

model = load_model_from_config(config, f"/storage/workspaces/artorg_aimi/ws_00000/lars/sd-v1-4.ckpt")
model = model.to(device)
model.eval()

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

class FileListDataset(Dataset):

    def __init__(self, file_list, labels, transform) -> None:
        self.transform = transform
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        img = Image.open(self.file_list[idx])
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


image_files = np.array(sorted(glob.glob('../drd_small/test/[012]/*/*.png')))

# Train images are x.png where x < 400
train_images = [i for i in image_files if int(i.split('/')[-1].split('.')[0]) < 400]
train_labels = [int(i.split('/')[-3]) for i in train_images]

test_images = [i for i in image_files if int(i.split('/')[-1].split('.')[0]) >= 400]
test_labels = [int(i.split('/')[-3]) for i in test_images]

train_data_in = FileListDataset(train_images, train_labels, transform=transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]))

print(f"Number of images: {len(train_data_in)}")

batch_size = 32

image_embeddings = []
labels = []
# Loop over training data
for cnt, (image, label) in enumerate(train_data_in):
    if cnt < args.offset:
        continue

    if cnt >= args.offset + args.num_images:
        break

    # Find class token
    tmp_token = get_model_attr(model, "cond_stage_model").tokenizer([class_names[label]], truncation=True, max_length=get_model_attr(model, "cond_stage_model").max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    
    # Re-initialize the trained embedding layer and optimizer. 
    get_model_attr(model, "cond_stage_model").transformer.text_model.embeddings.token_embedding.weight = nn.Parameter(original_embed.clone())
    get_model_attr(model, "cond_stage_model").transformer.text_model.embeddings.token_embedding.weight.requires_grad = True

    optimizer = torch.optim.Adam(get_model_attr(model, "cond_stage_model").transformer.text_model.embeddings.token_embedding.parameters(), lr=1e-2)

    # Wrap in DataParallel if more than 1 GPU is available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    losses = []

    tokens = tmp_token["input_ids"].to(device)
    original_id = tokens[0][1]

    with torch.no_grad():
        # Transform image
        image = image.unsqueeze(0).to(device)
        
        # Encode image to latent space
        z = get_model_attr(model, "encode_first_stage")(image).mode() * 0.18215

        # Duplicate the image to batch_size
        z = z.repeat(batch_size, 1, 1, 1)

    for epoch in range(args.iterations):
        # Embed the condition
        cc = get_model_attr(model, "cond_stage_model").forward_imgembs(prompts[label]).to(device).repeat(batch_size, 1, 1)

        # Noise latent space
        t = torch.randint(0, 50, size=(batch_size,), device=device) * 20 + 1
        noise = torch.randn_like(z)

        cumalphas_t = get_model_attr(model, "alphas_cumprod")[t][:, None, None, None]
        xt = torch.sqrt(cumalphas_t)*z + torch.sqrt(1 - cumalphas_t)*noise

        # Predict noise
        e_t = get_model_attr(model, "model").diffusion_model(xt, timesteps=t, context=cc)

        # Compute MSE loss
        optimizer.zero_grad()
        loss = ((e_t - noise)**2).mean() + 0.0 * (get_model_attr(model, "cond_stage_model").transformer.text_model.embeddings.token_embedding.weight[original_id] - anchors[label]).pow(2).sum()
        loss.backward()

        # Set the gradients for all tokens except the class token to zero
        get_model_attr(model, "cond_stage_model").transformer.text_model.embeddings.token_embedding.weight.grad[torch.arange(get_model_attr(model, "cond_stage_model").transformer.\
                                                                                                          text_model.embeddings.token_embedding.weight.grad.shape[0], device=device) != original_id] = 0

        optimizer.step()
        losses.append(loss.item())

    image_embedding = get_model_attr(model, "cond_stage_model").transformer.text_model.embeddings.token_embedding.weight[original_id].detach().cpu()
    image_embeddings.append(image_embedding)

    labels.append(label)

    if torch.cuda.device_count() > 1:
        model = model.module

    if cnt % 25 == 0:
        print(f"Last loss: {losses[-1]}. Processed {cnt} images", flush=True)
                        
image_embeds = torch.stack(image_embeddings).numpy()
print(image_embeds.shape)

labels = np.array(labels)

np.save(f"embed_drd_images_clip_{args.offset}_{args.num_images}.npy", image_embeds)
np.save(f"labels_drd_images_clip_{args.offset}_{args.num_images}.npy", labels)
