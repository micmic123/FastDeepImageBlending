# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:28:28 2019

@author: Owen and Tarmily
"""

from torch.nn import functional as F
import torch.nn as nn
from PIL import Image 
import numpy as np
from skimage.io import imsave
import cv2
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import models
from torchvision import utils as vutils
from torchvision import transforms
from collections import namedtuple
import pdb
import copy
import time
import random
from model import LaplacianFilter

import asyncio
import aiohttp
import async_timeout


def test(x_start, y_start, target_file, source_file, mask_file, args, outputdir, transfer, lf, vgg, mse, mean_shift, epoch, itr):
    # Hyperparameter Inputs
    gpu_id = args.device
    ss = args.ss  # source image size
    ts = args.ts  # target image size

    # Load target images
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        # transforms.Normalize(mean, std)
    ])
    target_img = Image.open(target_file).convert('RGB').resize((ts, ts))
    target_img = t(target_img).to(gpu_id).unsqueeze(0)
    source_img = Image.open(source_file).convert('RGB').resize((ss, ss))
    source_img = t(source_img).to(gpu_id).unsqueeze(0)
    mask = np.array(Image.open(mask_file).convert('L').resize((ss, ss)))
    mask[mask > 0] = 1
    mask = torch.from_numpy(mask).unsqueeze(0)

    # Import VGG network for computing style and content loss
    target_features_style = vgg(mean_shift(target_img))
    target_gram_style = [gram_matrix(y) for y in target_features_style]

    x = source_img

    # Make Canvas Mask
    canvas_mask = make_canvas_mask(x_start, y_start, target_img, mask)  # (B, ts, ts)
    canvas_mask = torch.from_numpy(canvas_mask).unsqueeze(1).float().to(gpu_id)  # (B, 1, ts, ts)

    # Compute gt_gradient
    gt_gradient = compute_gt_gradient(lf, canvas_mask, x_start, y_start, x, target_img, mask, gpu_id)  # list of (B, ts, ts)

    x_ts = torch.zeros((x.shape[0], x.shape[1], ts, ts)).to(gpu_id)  # (B, 3, ts, ts)
    x_ts[:, :, x_start - ss // 2:x_start + ss // 2, y_start - ss // 2:y_start + ss // 2] = x
    with torch.no_grad():
        input_img = transfer(torch.cat([canvas_mask, x_ts, target_img], dim=1))

    # Composite Foreground and Background to Make Blended Image
    canvas_mask = canvas_mask.expand((-1, 3, -1, -1))  # (B, 3, ts, ts)
    blend_img = torch.zeros(target_img.shape).to(gpu_id)
    blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1)  # I_B

    # Compute Laplacian Gradient of Blended Image
    pred_gradient = lf(blend_img)  # list of (B, ts, ts)

    # Compute Gradient Loss
    grad_loss = 0
    for c in range(len(pred_gradient)):
        grad_loss += mse(pred_gradient[c], gt_gradient[c])
    grad_loss /= len(pred_gradient)

    # Compute Style Loss
    blend_features_style = vgg(mean_shift(input_img))
    blend_gram_style = [gram_matrix(y) for y in blend_features_style]

    style_loss = 0
    for layer in range(len(blend_gram_style)):
        style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
    style_loss /= len(blend_gram_style)

    # Compute Content Loss
    blend_obj = blend_img[:, :, int(x_start - x.shape[2] * 0.5):int(x_start + x.shape[2] * 0.5),
                int(y_start - x.shape[3] * 0.5):int(y_start + x.shape[3] * 0.5)]
    m = mask.unsqueeze(1).expand((-1, 3, -1, -1)).to(gpu_id)
    source_object_features = vgg(mean_shift(x * m))
    blend_object_features = vgg(mean_shift(blend_obj * m))
    content_loss = mse(blend_object_features.relu2_2, source_object_features.relu2_2)

    # Compute TV Reg Loss
    tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
              torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))

    print(f'grad : {grad_loss.item():.4f}, style : {style_loss.item():.4f}, '
          f'content: {content_loss.item():.4f}, tv: {tv_loss.item():.4f}')

    blend_img = blend_img.detach()
    blend_img.data.clamp_(0, 255)
    input_img = input_img.detach()
    input_img.data.clamp_(0, 255)
    out = torch.cat([x_ts, x_ts * canvas_mask, blend_img, input_img], dim=0)
    save_grid(out, f'{outputdir}/test_{epoch:02}_{itr:04}_{grad_loss.item():.4f}_{style_loss.item():.4f}_'
                   f'{content_loss.item():.4f}.png', nrow=1)


def _prepare(img):
    return img.permute((1,2,0)).numpy().astype(np.uint8)


def save_img(tensor, path):
    # tensor: (1, 3, H, W)
    img = _prepare(tensor.squeeze(0).cpu())
    Image.fromarray(img).save(path)


def save_grid(tensor, path, nrow=1):
    grid = vutils.make_grid(tensor.cpu(), nrow=nrow)
    img = _prepare(grid)
    Image.fromarray(img).save(path)


def numpy2tensor(np_array, gpu_id):
    if len(np_array.shape) == 2:
        tensor = torch.from_numpy(np_array).unsqueeze(0).float().to(gpu_id)
    else:
        tensor = torch.from_numpy(np_array).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    return tensor


def make_canvas_mask(x_start, y_start, target_img, mask):
    canvas_mask = np.zeros((mask.shape[0], target_img.shape[2], target_img.shape[3]))
    canvas_mask[:, int(x_start-mask.shape[1]*0.5):int(x_start+mask.shape[1]*0.5),
                   int(y_start-mask.shape[2]*0.5):int(y_start+mask.shape[2]*0.5)] = mask
    return canvas_mask

def laplacian_filter_tensor(img_tensor, gpu_id):

    laplacian_filter = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    laplacian_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    laplacian_conv.weight = nn.Parameter(torch.from_numpy(laplacian_filter).float().unsqueeze(0).unsqueeze(0).to(gpu_id))
    
    for param in laplacian_conv.parameters():
        param.requires_grad = False
    
    red_img_tensor = img_tensor[:,0,:,:].unsqueeze(1)
    green_img_tensor = img_tensor[:,1,:,:].unsqueeze(1)
    blue_img_tensor = img_tensor[:,2,:,:].unsqueeze(1)
    
    red_gradient_tensor = laplacian_conv(red_img_tensor).squeeze(1) 
    green_gradient_tensor = laplacian_conv(green_img_tensor).squeeze(1) 
    blue_gradient_tensor = laplacian_conv(blue_img_tensor).squeeze(1)
    return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor
    

def compute_gt_gradient(lf, canvas_mask, x_start, y_start, source_img, target_img, mask, gpu_id):
    canvas_mask = canvas_mask[:, 0].cpu().numpy()  # (B, ts, ts)
    mask = mask.cpu().numpy()
    # compute source image gradient
    # source_img_tensor = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    source_img_tensor = source_img
    red_source_gradient_tensor, green_source_gradient_tensor, blue_source_gradient_tenosr = lf(source_img_tensor)  # (B, ss, ss)
    red_source_gradient = red_source_gradient_tensor.cpu().data.numpy()  # (B, ss, ss)
    green_source_gradient = green_source_gradient_tensor.cpu().data.numpy()
    blue_source_gradient = blue_source_gradient_tenosr.cpu().data.numpy()
    
    # compute target image gradient
    # target_img_tensor = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    target_img_tensor = target_img
    red_target_gradient_tensor, green_target_gradient_tensor, blue_target_gradient_tenosr = lf(target_img_tensor)  # (B, ts, ts)
    red_target_gradient = red_target_gradient_tensor.cpu().data.numpy()
    green_target_gradient = green_target_gradient_tensor.cpu().data.numpy()
    blue_target_gradient = blue_target_gradient_tenosr.cpu().data.numpy()

    # foreground gradient
    red_source_gradient = red_source_gradient * mask
    green_source_gradient = green_source_gradient * mask
    blue_source_gradient = blue_source_gradient * mask
    red_foreground_gradient = np.zeros((canvas_mask.shape))
    red_foreground_gradient[:, int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)] = red_source_gradient
    green_foreground_gradient = np.zeros((canvas_mask.shape))
    green_foreground_gradient[:, int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)] = green_source_gradient
    blue_foreground_gradient = np.zeros((canvas_mask.shape))
    blue_foreground_gradient[:, int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)] = blue_source_gradient
    
    # background gradient
    red_background_gradient = red_target_gradient * (canvas_mask - 1) * (-1)
    green_background_gradient = green_target_gradient * (canvas_mask - 1) * (-1)
    blue_background_gradient = blue_target_gradient * (canvas_mask - 1) * (-1)
    
    # add up foreground and background gradient
    gt_red_gradient = red_foreground_gradient + red_background_gradient
    gt_green_gradient = green_foreground_gradient + green_background_gradient
    gt_blue_gradient = blue_foreground_gradient + blue_background_gradient
    
    gt_red_gradient = torch.from_numpy(gt_red_gradient).float().to(gpu_id)  # (B, ts, ts)
    gt_green_gradient = torch.from_numpy(gt_green_gradient).float().to(gpu_id)  # (B, ts, ts)
    gt_blue_gradient = torch.from_numpy(gt_blue_gradient).float().to(gpu_id)  # (B, ts, ts)
    
    gt_gradient = [gt_red_gradient, gt_green_gradient, gt_blue_gradient]
    return gt_gradient  # list of (B, ts, ts)




class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std



class MeanShift(nn.Conv2d):
    def __init__(self, gpu_id):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        rgb_range=1
        rgb_mean=(0.4488, 0.4371, 0.4040)
        rgb_std=(1.0, 1.0, 1.0)
        sign=-1
        std = torch.Tensor(rgb_std).to(gpu_id)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).to(gpu_id) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).to(gpu_id) / std
        for p in self.parameters():
            p.requires_grad = False


def get_matched_features_numpy(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False)
    cpu_blended_features = blended_features.cpu().detach().numpy()
    cpu_target_features = target_features.cpu().detach().numpy()
    for filter in range(0, blended_features.size(1)):
        matched_filter = torch.from_numpy(hist_match_numpy(cpu_blended_features[0, filter, :, :],
                                                           cpu_target_features[0, filter, :, :])).to(blended_features.device)
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def get_matched_features_pytorch(blended_features, target_features):
    matched_features = blended_features.new_full(size=blended_features.size(), fill_value=0, requires_grad=False).to(blended_features.device)
    for filter in range(0, blended_features.size(1)):
        matched_filter = hist_match_pytorch(blended_features[0, filter, :, :], target_features[0, filter, :, :])
        matched_features[0, filter, :, :] = matched_filter
    return matched_features


def hist_match_pytorch(source, template):

    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    return matched_source.reshape(oldshape)


async def hist_match_pytorch_async(source, template, index, storage):

    oldshape = source.size()
    source = source.view(-1)
    template = template.view(-1)

    max_val = max(source.max().item(), template.max().item())
    min_val = min(source.min().item(), template.min().item())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        storage[0, index, :, :] = source.reshape(oldshape)
        return

    hist_bin_centers = torch.arange(start=min_val, end=max_val, step=hist_step).to(source.device)
    hist_bin_centers = hist_bin_centers + hist_step / 2.0

    source_hist = torch.histc(input=source, min=min_val, max=max_val, bins=num_bins)
    template_hist = torch.histc(input=template, min=min_val, max=max_val, bins=num_bins)

    source_quantiles = torch.cumsum(input=source_hist, dim=0)
    source_quantiles = source_quantiles / source_quantiles[-1]

    template_quantiles = torch.cumsum(input=template_hist, dim=0)
    template_quantiles = template_quantiles / template_quantiles[-1]

    nearest_indices = torch.argmin(torch.abs(template_quantiles.repeat(len(source_quantiles), 1) - source_quantiles.view(-1, 1).repeat(1, len(template_quantiles))), dim=1)

    source_bin_index = torch.clamp(input=torch.round(source / hist_step), min=0, max=num_bins - 1).long()

    mapped_indices = torch.gather(input=nearest_indices, dim=0, index=source_bin_index)
    matched_source = torch.gather(input=hist_bin_centers, dim=0, index=mapped_indices)

    storage[0, index, :, :] = matched_source.reshape(oldshape)


async def loop_features_pytorch(source, target, storage):
    size = source.shape
    tasks = []

    for i in range(0, size[1]):
        task = asyncio.ensure_future(hist_match_pytorch_async(source[0, i], target[0, i], i, storage))
        tasks.append(task)

    await asyncio.gather(*tasks)


def get_matched_features_pytorch_async(source, target, matched):
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(loop_features_pytorch(source, target, matched))
    loop.run_until_complete(future)
    loop.close()


def hist_match_numpy(source, template):

    oldshape = source.shape

    source = source.ravel()
    template = template.ravel()

    max_val = max(source.max(), template.max())
    min_val = min(source.min(), template.min())

    num_bins = 400
    hist_step = (max_val - min_val) / num_bins

    if hist_step == 0:
        return source.reshape(oldshape)

    source_hist, source_bin_edges = np.histogram(a=source, bins=num_bins, range=(min_val, max_val))
    template_hist, template_bin_edges = np.histogram(a=template, bins=num_bins, range=(min_val, max_val))

    hist_bin_centers = source_bin_edges[:-1] + hist_step / 2.0

    source_quantiles = np.cumsum(source_hist).astype(np.float32)
    source_quantiles /= source_quantiles[-1]
    template_quantiles = np.cumsum(template_hist).astype(np.float32)
    template_quantiles /= template_quantiles[-1]

    index_function = np.vectorize(pyfunc=lambda x: np.argmin(np.abs(template_quantiles - x)))

    nearest_indices = index_function(source_quantiles)

    source_data_bin_index = np.clip(a=np.round(source / hist_step), a_min=0, a_max=num_bins-1).astype(np.int32)

    mapped_indices = np.take(nearest_indices, source_data_bin_index)
    matched_source = np.take(hist_bin_centers, mapped_indices)

    return matched_source.reshape(oldshape)


def main():
    size = (64, 512, 512)
    source = np.random.randint(low=0, high=500000, size=size).astype(np.float32)
    target = np.random.randint(low=0, high=500000, size=size).astype(np.float32)
    source_tensor = torch.Tensor(source).to(0)
    target_tensor = torch.Tensor(target).to(0)
    matched_numpy = np.zeros(shape=size)
    matched_pytorch = torch.zeros(size=size, device=0)

    numpy_time = time.process_time()

    for i in range(0, size[0]):
        matched_numpy[i, :, :] = hist_match_numpy(source[i], target[i])
    
    numpy_time = time.process_time() - numpy_time

    pytorch_time = time.process_time()

    for i in range(0, size[0]):
        matched_pytorch[i, :, :] = hist_match_pytorch(source_tensor[i], target_tensor[i])
    
    pytorch_time = time.process_time() - pytorch_time


if __name__ == "__main__":
    main()
