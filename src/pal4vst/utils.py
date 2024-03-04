from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from PIL import Image
import numpy as np 
import torch
import cv2
import pdb
import os
import glob


def get_paths(path):
    if os.path.isfile(path):
        return [path]
    # should be a directory then
    return sorted(glob.glob(path + '/*'))

def get_mean_stdinv(img):
    """
    Compute the mean and std for input image (make sure it's aligned with training)
    """

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    mean_img = np.zeros((img.shape))
    mean_img[:,:,0] = mean[0]
    mean_img[:,:,1] = mean[1]
    mean_img[:,:,2] = mean[2]
    mean_img = np.float32(mean_img)

    std_img = np.zeros((img.shape))
    std_img[:,:,0] = std[0]
    std_img[:,:,1] = std[1]
    std_img[:,:,2] = std[2]
    std_img = np.float64(std_img)

    stdinv_img = 1 / np.float32(std_img)

    return mean_img, stdinv_img

def numpy2tensor(img):
    """
    Convert numpy to tensor
    """
    img = torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()
    return img

def prepare_input(img, device=None):
    if device is None: device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(img, Image.Image): 
        img = img.resize((512, 512))
        img = np.asarray(img)
    """
    Convert numpy image into a normalized tensor (ready to do segmentation)
    """
    mean_img, stdinv_img = get_mean_stdinv(img)
    img_tensor = numpy2tensor(img).to(device)
    mean_img_tensor = numpy2tensor(mean_img).to(device)
    stdinv_img_tensor = numpy2tensor(stdinv_img).to(device)
    img_tensor = img_tensor - mean_img_tensor
    img_tensor = img_tensor * stdinv_img_tensor
    return img_tensor

# FIXME use opencv to draw text
def draw_text_on_image(image_path, text, scale):
    # Open an image file
    with Image.open(image_path) as img:
        width, height = img.size

        # Determine the size of the text based on the image size and scale
        text_size = int(min(width, height) * scale)
        # Use a truetype font from PIL, adjust the path to the font file as needed
        # Here we're using the DejaVuSans which is typically installed with matplotlib
        font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Determine text position
        text_position = (20, 0) # horizontal, vertical
        # Add text to image
        draw.text(text_position, text, font=font, fill=(255, 105, 180))

    return img

class PAL4VST:
    IMAGE_DIMENSIONS = (512, 512)
    def __init__(self,
                 torchscript_file = './deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt',
                 device=None
                 ):
        if device is None: self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = torch.load(torchscript_file).to(self.device)
    def resize(self, img :Image)->Image:
        return img.resize(self.IMAGE_DIMENSIONS)
    def numpy_to_tensor(self, img):
        """
        Convert numpy to tensor
        """
        img = torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()
        return img
    def img_to_tensor(self, img: Image):
        img = self.resize(img)
        img = np.asarray(img)
        """
        Convert numpy image into a normalized tensor (ready to do segmentation)
        """
        mean_img, stdinv_img = self.get_mean_stdinv(img)
        img_tensor = self.numpy_to_tensor(img).to(self.device)
        mean_img_tensor = self.numpy_to_tensor(mean_img).to(self.device)
        stdinv_img_tensor = self.numpy_to_tensor(stdinv_img).to(self.device)
        img_tensor = img_tensor - mean_img_tensor
        img_tensor = img_tensor * stdinv_img_tensor
        return img_tensor
    def get_mean_stdinv(self,img):
        """
        Compute the mean and std for input image (make sure it's aligned with training)
        """

        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]

        mean_img = np.zeros((img.shape))
        mean_img[:,:,0] = mean[0]
        mean_img[:,:,1] = mean[1]
        mean_img[:,:,2] = mean[2]
        mean_img = np.float32(mean_img)

        std_img = np.zeros((img.shape))
        std_img[:,:,0] = std[0]
        std_img[:,:,1] = std[1]
        std_img[:,:,2] = std[2]
        std_img = np.float64(std_img)

        stdinv_img = 1 / np.float32(std_img)

        return mean_img, stdinv_img
    def pal(self, img_tensor):
        return self.model(img_tensor)
    def pal_np(self, img_tensor):
        return self.pal(img_tensor).cpu().data.numpy()[0][0]

    def overlay(self, img: Image, pal_np, color=(255,0,255), alpha=0.3):
        overlay = np.zeros((img.shape)); overlay[:,:,0] = color[0]; overlay[:,:,1] = color[1], overlay[:,:,2] = color[2]
        img_with_pal = img * (1 - pal_np[:,:,None]) + alpha * overlay * pal_np[:,:,None] + (1 - alpha) * img * pal_np[:,:,None]
        image = Image.fromarray(img_with_pal.astype(np.uint8))
        return image

    def par(self, pal_np) ->float:
        return pal_np.sum() / (pal_np.shape[0] * pal_np.shape[1])

    def draw_text_on_img(self,
                         img: Image,
                         text: str,
                         scale: float=1,
                         position=(20,0),
                         color=(255,0,255) # BGR Format for opencv
                         ):
        np_image = np.array(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1 * scale
        font_thickness = 2 * scale
        cv2.putText(np_image, text, position, font, font_scale, color, font_thickness)
        return Image.fromarray(np_image)



    def get(self, img: Image):
        pal_np = self.pal_np(self.img_to_tensor(img))
        par = self.par(pal_np)
        overlay = self.overlay(img, pal_np)
        return par, overlay

