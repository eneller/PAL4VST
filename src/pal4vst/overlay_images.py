from diffusers import StableDiffusionXLInpaintPipeline
from skimage.io import imsave
from PIL import Image
import numpy as np 
from utils import *
import argparse
import torch
import cv2
import pdb
import os 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='./demo_test_data/stylegan2_ffhq/images')
    parser.add_argument('--output', type=str, default='./demo_results/stylegan2_ffhq_rank')
    parser.add_argument('--torchscript_file', type=str, default='./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.3)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_refine_file), exist_ok = True)
    
    # load PAl model
    model = torch.load(args.torchscript_file).to(args.device)

    for img_file in tqdm(get_imgs(args.input_img_dir)):
        # compute pal on the generated image
        img = np.array(Image.open(img_file).resize((512, 512)))    
        img_tensor = prepare_input(img, args.device)
        pal = model(img_tensor)
        pal_np = pal.cpu().data.numpy()[0][0]    
        pink = np.zeros((img.shape)); pink[:,:,0] = 255; pink[:,:,2] = 255
        img_with_pal = img * (1 - pal_np[:,:,None]) + args.alpha * pink * pal_np[:,:,None] + (1 - args.alpha) * img * pal_np[:,:,None]
        image = Image.fromarray(img_with_pal.astype(np.uint8))
        image.save(os.path.join(args.output), os.path.basename(img_file))
    
