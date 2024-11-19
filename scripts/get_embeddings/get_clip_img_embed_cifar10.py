# -*- coding: utf-8 -*-
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torchvision

from transformers import CLIPProcessor, CLIPModel


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# generation hyperparameters.
parser.add_argument('--gaussian_mag_ood_det', type=float, default=0.07)
args = parser.parse_args()

train_data_in = torchvision.datasets.CIFAR10("../data/", train=True)

num_classes = 10
sum_temp = 0

# Get CLIP embeddings for CIFAR-10 images
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Count model params
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

model = model.cuda()

batch_size = 200

# CIFAR10 label texts
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_texts = [f"A high-quality image of a{'n' if name[0] in 'aeiou' else ' '} {name}" for name in class_names]

image_embeds = []
labels = []
# Loop over training data in batches. TODO: Fails if batch_size does not divide len(train_data_in)
assert len(train_data_in) % batch_size == 0
with torch.no_grad():
    img_batch = []
    for cnt, (image, label) in enumerate(train_data_in):
        img_batch.append(image)
        labels.append(label)

        if len(img_batch) < batch_size:
            continue
        
        image = img_batch
        img_batch = []
            
        inputs = processor(text=label_texts, images=image, return_tensors="pt", padding=True)
        for k,v in inputs.items():
            inputs[k] = v.cuda()

        image_embs = model.encode_image(inputs["pixel_values"])
        print(image_embs.shape)
        image_embeds.append(image_embs)

        if cnt % 25 == 0:
            print(f"Processed {cnt} images")
                        
text_embeds = model.encode_text(inputs["input_ids"].cuda())
image_embeds = torch.cat(image_embeds, dim=0)
labels = np.array(labels)
print(image_embeds.shape, text_embeds.shape)

# Verify image embeddings align with the text embeddings
normalized_anchors = F.normalize(text_embeds, p=2, dim=1)
for i in range(25):
    print(train_data_in[i][1], F.cosine_similarity(F.normalize(image_embeds[i].unsqueeze(0), p=2, dim=1), normalized_anchors).argmax())

for index in range(10):
    sum_temp += 5000  # number_dict[index]
# breakpoint()
if sum_temp == num_classes * 5000:
    for index in range(num_classes):
        print(index)
        
        class_embs = image_embeds[labels == index]
        normalized_class_embs = F.normalize(class_embs, p=2, dim=1)
        class_norms = class_embs.norm(dim=1)

        new_dis = MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())

        negative_samples = new_dis.rsample((len(class_embs),))
        negative_samples = negative_samples * args.gaussian_mag_ood_det
        
        sample_point = normalized_class_embs + negative_samples
        
        if index == 0:
            ood_samples = [sample_point * class_norms.unsqueeze(1)]
        else:
            ood_samples.append(sample_point * class_norms.unsqueeze(1))

print(torch.stack(ood_samples).cpu().data.numpy().shape)

np.save \
        ('./cifar10_outlier_img_clip_noise_' + str(args.gaussian_mag_ood_det) + '.npy', torch.stack(ood_samples).cpu().data.numpy())
