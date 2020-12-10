import os
import numpy as np
from PIL import Image
import argparse
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve


parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/1_source.png', help='path to the source image')
parser.add_argument('--mask_file', type=str, default='data/1_mask.png', help='path to the source mask image')
parser.add_argument('--target_file', type=str, default='data/1_target.png', help='path to the target image')
parser.add_argument('--preset', type=int, default=None, help='preset for test [0, 5]')
parser.add_argument('--ss', type=int, default=300, help='source image size')
parser.add_argument('--ts', type=int, default=512, help='target image size')
parser.add_argument('--x', type=int, default=200, help='vertical location 240')
parser.add_argument('--y', type=int, default=235, help='vertical location 256')
args = parser.parse_args()

presets = {
    0: (240, 350),
    1: (200, 235),
    4: (340, 320),
    5: (150, 255),
}


def laplacian_matrix(n, m):
    """Generate the Poisson matrix.
    Refer to:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    Note: it's the transpose of the wiki's matrix
    Code from: https://github.com/PPPW/poisson-image-editing
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)

    return mat_A


def make_canvas_mask_numpy(x_start, y_start, target_img, mask):
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1], 3))
    canvas_mask[int(x_start-mask.shape[0]*0.5):int(x_start+mask.shape[0]*0.5),
                   int(y_start-mask.shape[1]*0.5):int(y_start+mask.shape[1]*0.5), :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return canvas_mask


def make_canvas_mask_grayscale(x_start, y_start, target_img, mask):
    canvas_mask = np.zeros((target_img.shape[0], target_img.shape[1]))
    canvas_mask[int(x_start-mask.shape[0]*0.5):int(x_start+mask.shape[0]*0.5),
                   int(y_start-mask.shape[1]*0.5):int(y_start+mask.shape[1]*0.5)] = mask
    return canvas_mask


def image_concat(source_image, mask_image, target_image, x_start, y_start):
    '''
    c&p image blending of source image with mask image(In pillow format) to target
    image. places it at [x_position, y_position]. returns pillow format.
    '''

    source_image = np.array(source_image)
    mask_image = np.array(mask_image)
    target_image = np.array(target_image)
    ts = target_image.shape[0]
    ss = source_image.shape[0]

    canvas_mask = make_canvas_mask_numpy(x_start, y_start, target_image, mask_image)
    canvas_mask = (canvas_mask - canvas_mask.min()) / (canvas_mask.max() - canvas_mask.min())

    x_ts = np.zeros_like(target_image)
    x_ts[x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2, :] = source_image

    blend_img = np.zeros_like(target_image)
    blend_img = x_ts * canvas_mask + target_image * (1 - canvas_mask)
    blend_img = blend_img.astype(np.uint8)
    return blend_img


def image_alpha(source_image, mask_image, target_image, x_start, y_start):
    '''
    c&p image blending of source image with mask image(In pillow format) to target
    image. places it at [x_position, y_position]. returns pillow format.
    gaussian blur the mask image by sigma=3
    '''
    source_image = np.array(source_image)
    mask_image = np.array(mask_image)
    target_image = np.array(target_image)
    ts = target_image.shape[0]
    ss = source_image.shape[0]

    canvas_mask = make_canvas_mask_numpy(x_start, y_start, target_image, mask_image)
    canvas_mask = cv2.GaussianBlur(canvas_mask, (9,9), 3)
    canvas_mask = (canvas_mask - canvas_mask.min()) / (canvas_mask.max() - canvas_mask.min())

    x_ts = np.zeros_like(target_image)
    x_ts[x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2, :] = source_image

    blend_img = np.zeros_like(target_image)
    blend_img = x_ts * canvas_mask + target_image * (1 - canvas_mask)
    blend_img = blend_img.astype(np.uint8)
    return blend_img


def image_poisson(source_image, mask_image, target_image, x_start, y_start):
    '''
    image blending of source image with mask image(In pillow format) to target
    image. places it at [x_position, y_position]. returns pillow format.
    poisson image editing
    Original Reference: https://github.com/PPPW/poisson-image-editing
    Slightly edited to fit our need(comparison)
    '''
    source_image = np.array(source_image)
    mask_image = np.array(mask_image)
    target_image = np.array(target_image)
    ts = target_image.shape[0]
    ss = source_image.shape[0]

    # blend_img = cv2.seamlessClone(source_image, target_image, mask_image, (y_start, x_start), cv2.NORMAL_CLONE)

    canvas_mask = make_canvas_mask_grayscale(x_start, y_start, target_image, mask_image)
    canvas_mask[canvas_mask != 0] = 1
    x_ts = np.zeros_like(target_image)
    x_ts[x_start - ss//2:x_start + ss//2, y_start - ss//2:y_start + ss//2, :] = source_image

    blend_img = np.zeros_like(target_image)


    mat_A = laplacian_matrix(ts, ts)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity
    for y in range(1, ts - 1):
        for x in range(1, ts - 1):
            if canvas_mask[y, x] == 0:
                k = x + y * ts
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + ts] = 0
                mat_A[k, k - ts] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = canvas_mask.flatten()
    blend_img = np.zeros_like(target_image)
    for channel in range(source_image.shape[2]): # For RGB
        source_flat = x_ts[:, :, channel].flatten()
        target_flat = target_image[:, :, channel].flatten()

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((ts, ts))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        blend_img[:, :, channel] = x

    return blend_img


def save(img, path):
    Image.fromarray(img).save(path)


def main():
    target = args.target_file
    source = args.source_file
    mask = args.mask_file
    ss = args.ss
    ts = args.ts
    x_start = args.x
    y_start = args.y

    if args.preset is not None:
        assert 0 <= args.preset <= 5
        target = f'./data/{args.preset}_target.png'
        source = f'./data/{args.preset}_source.png'
        mask = f'./data/{args.preset}_mask.png'
        x_start, y_start = presets[args.preset]

    target_img = Image.open(target).convert('RGB').resize((ts, ts))
    mask_img = np.array(Image.open(mask).convert('L').resize((ss, ss)))
    mask_img[mask_img > 0] = 1
    mask_img = Image.fromarray(mask_img)
    source = Image.open(source).convert('RGB').resize((ss, ss))

    concat = image_concat(source,mask_img,target_img, x_start, y_start)
    alpha = image_alpha(source,mask_img,target_img, x_start, y_start)
    poisson = image_poisson(source, mask_img, target_img, x_start, y_start)

    os.makedirs('./results_trad', exist_ok=True)
    save(concat, f'results_trad/{os.path.basename(target).split("_")[0]}_naive_{x_start}-{y_start}.png')
    save(alpha, f'results_trad/{os.path.basename(target).split("_")[0]}_alpha_{x_start}-{y_start}.png')
    save(poisson, f'results_trad/{os.path.basename(target).split("_")[0]}_poisson_{x_start}-{y_start}.png')


if __name__ == "__main__":
    main()
