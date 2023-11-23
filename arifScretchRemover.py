import gradio as gr
import numpy as np
import torch
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from scratch_detection import ScratchDetection

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DEISMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import time
import os

device = "cuda"


def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)


def generate_scratch_mask():
    # Save the input image to a directory
    img_p = "auny.png"
    input_image = PIL.Image.open(img_p).convert('RGB')

    input_image_path = ("input_images/%s" % img_p)
    input_image_resized = resize_image(input_image, 768)
    input_image_resized.save(input_image_path)

    test_path = "input_images"
    output_dir = "output_masks"
    scratch_detector = ScratchDetection(test_path, output_dir, input_size="scale_256", gpu=0)
    scratch_detector.run()
    mask_image = scratch_detector.get_mask_image(img_p)

    # Resize the mask to match the input image size
    mask_image = mask_image.resize(input_image.size, Image.BICUBIC)

    # Apply dilation to make the lines bigger
    kernel = np.ones((5, 5), np.uint8)
    mask_image_np = np.array(mask_image)
    mask_image_np_dilated = cv2.dilate(mask_image_np, kernel, iterations=2)
    mask_image_dilated = Image.fromarray(mask_image_np_dilated)



    # window_name = 'Output Image'
    # cv2.imshow(window_name, mask_image_dilated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # return mask_image_dilated


generate_scratch_mask()