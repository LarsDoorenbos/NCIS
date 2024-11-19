# -*- coding: utf-8 -*-
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# generation hyperparameters.
parser.add_argument('--gaussian_mag_ood_det', type=float, default=0.07)
args = parser.parse_args()


anchor = torch.from_numpy(np.load('./token_embed_c10.npy')).cuda()
print(anchor.shape)
num_classes = 10
sum_temp = 0

for index in range(10):
    sum_temp += 5000  # number_dict[index]
# breakpoint()
if sum_temp == num_classes * 5000:
    for index in range(num_classes):
        ID = F.normalize(anchor[index].unsqueeze(0), p=2, dim=1)
        print(index)
        
        new_dis = MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())

        negative_samples = new_dis.rsample((1000,))
        negative_samples = negative_samples * args.gaussian_mag_ood_det
        
        sample_point = ID + negative_samples
        
        if index == 0:
            ood_samples = [sample_point * anchor[index].norm()]
        else:
            ood_samples.append(sample_point * anchor[index].norm())

print(torch.stack(ood_samples).cpu().data.numpy().shape)

np.save \
        ('./cifar10_outlier_clip_noise_' + str(args.gaussian_mag_ood_det) + '.npy', torch.stack(ood_samples).cpu().data.numpy())
