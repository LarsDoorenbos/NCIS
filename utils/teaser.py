
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F

import sys, platform

if platform.node() == 'lars-HP-ENVY-Laptop-15-ep0xxx':
    sys.path.append('/home/lars/Outliers/nonlinear-outlier-synthesis/scripts')
else:
    sys.path.append('/storage/homefs/ld20d359/nonlinear-outlier-synthesis/scripts')

from dream_ood import get_class_names

class Object(object):
    pass



seed = np.random.randint(0, 10000)
seed = 5465
print(f'Seed: {seed}')  
np.random.seed(seed)

names = ['Dream-OOD', 'Ours']

opt = Object()

dirs = [
    '/home/lars/generated_images_c100/samples'
]
opt.id_data = 'c100'

# dirs = [
#     '/home/lars/generated_images_in100/samples'
# ]
# opt.id_data = 'in100'


num_classes = 3
num_images = 4

# get 5 random classes
classes = np.random.choice(100, num_classes, replace=False)
classes = [99, 4, 9]

# get 3 random images from each class
fig, axs = plt.subplots(num_classes, num_images, figsize=(num_images * 4, num_classes * 4))
for cnt, cl in enumerate(classes):
    imgs = glob.glob(os.path.join(dirs[0], f'{cl}/*.png'))
    imgs = np.random.choice(imgs, num_images, replace=False)
    for i, img in enumerate(imgs):
        img = plt.imread(img)
        
        axs[cnt, i].imshow(img)
        # axs[cnt, i].axis('off')
        axs[cnt, i].spines['top'].set_visible(False)
        axs[cnt, i].spines['right'].set_visible(False)
        axs[cnt, i].spines['bottom'].set_visible(False)
        axs[cnt, i].spines['left'].set_visible(False)
        axs[cnt, i].set_xticks([])
        axs[cnt, i].set_yticks([])

    name = get_class_names(opt)[cl]
    # Change first letter to uppercase
    name = name[0].upper() + name[1:]

    axs[cnt, 0].set_ylabel(f'{name}', fontsize=20, fontweight='bold')

plt.subplots_adjust(wspace=0.05, hspace=0.01)

plt.savefig(f'teaser.png', bbox_inches='tight', pad_inches=0.0)