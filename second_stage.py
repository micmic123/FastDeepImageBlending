# Packages
import os
import numpy as np
import torch
from L_BFGS import LBFGS
from PIL import Image
from skimage.io import imsave
from utils import MeanShift, Vgg16, gram_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/our_first_pass.png', help='path to the source image')
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='path to the target image')
parser.add_argument('--preset', type=int, default=None, help='preset for test [0, 5]')
parser.add_argument('--ss', type=int, default=300, help='source image size')
parser.add_argument('--ts', type=int, default=512, help='target image size')
parser.add_argument('--x', type=int, default=200, help='vertical location')
parser.add_argument('--y', type=int, default=235, help='vertical location')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of iterations in each pass')
opt = parser.parse_args()

basedir = './results_second'
os.makedirs(basedir, exist_ok=True)

# Inputs
first_pass_img_file = opt.source_file
target_file = opt.target_file
if opt.preset is not None:
    assert 0 <= opt.preset <= 5
    target_file = f'data/{opt.preset}_target.png'

# Hyperparameter Inputs
gpu_id = opt.gpu_id
num_steps = opt.num_steps
ss = opt.ss  # source image size
ts = opt.ts  # target image size
x_start = opt.x
y_start = opt.y  # blending location

# Define Loss Functions
mse = torch.nn.MSELoss()

# Import VGG network for computing style and content loss
mean_shift = MeanShift(gpu_id)
vgg = Vgg16().to(gpu_id)

###################################
########### Second Pass ###########
###################################

# Default weights for loss functions in the second pass
style_weight = 1e7
# content_weight = 1
tv_weight = 1e-6
ss = 512
ts = 512
num_steps = opt.num_steps
first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))
target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(gpu_id)
target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(gpu_id)


# Define LBFGS optimizer
def get_input_optimizer(first_pass_img):
    optimizer = LBFGS([first_pass_img.requires_grad_()])
    return optimizer


optimizer = get_input_optimizer(first_pass_img)
print('Optimizing...')
run = [0]
while run[0] <= num_steps:
    def closure():
        # Compute Loss Loss    
        target_features_style = vgg(mean_shift(target_img))
        target_gram_style = [gram_matrix(y) for y in target_features_style]
        blend_features_style = vgg(mean_shift(first_pass_img))
        blend_gram_style = [gram_matrix(y) for y in blend_features_style]
        style_loss = 0
        for layer in range(len(blend_gram_style)):
            style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
        style_loss /= len(blend_gram_style)

        # Compute Total Loss and Update Image
        loss = style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()

        # Print Loss
        if run[0] % 1 == 0:
            print("run {}: ".format(run[0]), end='')
            print(' style : {:4f}'.format(style_loss.item()))
        if run[0] % 100 == 0:
            r = torch.clamp(first_pass_img.detach().cpu(), 0, 255)
            r = r.transpose(1, 3).transpose(1, 2).cpu().data.numpy()[0]
            imsave(os.path.join(basedir, os.path.basename(first_pass_img_file).split('.')[0] + f'_{run[0]:04}.png'),
                   r.astype(np.uint8))

        run[0] += 1
        return loss
    optimizer.step(closure)

