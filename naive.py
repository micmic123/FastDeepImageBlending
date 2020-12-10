import os
import numpy as np
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/1_source.png', help='path to the source image')
parser.add_argument('--mask_file', type=str, default='data/1_mask.png', help='path to the source mask image')
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='path to the target image')
parser.add_argument('--ss', type=int, default=300, help='source image size')
parser.add_argument('--ts', type=int, default=512, help='target image size')
parser.add_argument('--x', type=int, default=200, help='vertical location 240')
parser.add_argument('--y', type=int, default=235, help='vertical location 256')
args = parser.parse_args()

target_file = args.target_file
source_file = args.source_file
mask_file = args.mask_file
ss = args.ss
ts = args.ts
x_start = args.x
y_start = args.y

target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
source_img = np.array(Image.open(source_file).convert('RGB').resize((ss, ss)))
mask = np.array(Image.open(mask_file).convert('L').resize((ss, ss)))
mask[mask > 0] = 1
mask = np.expand_dims(mask, axis=-1)

target_region = target_img[x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2, :]
blend_region = source_img * mask + target_region * (1-mask)
target_img[x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2, :] = blend_region
Image.fromarray(target_img).save(f'results/{os.path.basename(target_file).split("_")[0]}_naive.png')
