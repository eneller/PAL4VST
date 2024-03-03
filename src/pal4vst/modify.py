# Modify Images in different ways for testing purposes.
# Functions can be called by their names from cli.
from tqdm import tqdm
from PIL import Image, ImageFilter

import argparse
import os

from utils import get_paths

args = None



def black(img: Image)-> Image:
    pass

def blur(img: Image)-> Image:
    return img.filter(ImageFilter.BLUR)

def main():
    global args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, default='./demo_test_data/stylegan2_ffhq/images')
    parser.add_argument('--output', type=str, default='./demo_results/stylegan2_ffhq_rank')
    parser.add_argument('--func', type=str, required=True)
    parser.add_argument('--x', nargs='+') # allow passing of other varying arguments
    args, unparsed= parser.parse_known_args() # alternatively use unparsed args?
    func = globals()[args.func]# hacky solution to call functions from strings

    for img_file in tqdm(get_paths(args.input)):
        img = Image.open(img_file)
        modified = func(img)
        outpath = os.path.join(args.output, os.path.basename(img_file))
        modified.save(outpath)

if __name__ == "__main__":
    main()