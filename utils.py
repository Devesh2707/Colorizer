import cv2
import io
import numpy as np
from skimage.color import rgb2lab, lab2rgb

import torch
from torchvision import transforms

from generator import load_weights

from google_drive_downloader import GoogleDriveDownloader as gdd

import config

def download_weights():
    gdd.download_file_from_google_drive(file_id="1pf62jJN-v6RpYEtwPsjh54STqO77UGz_",
                                        dest_path = config.MODEL_WEIGHTS)

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def preprocess(img, image_size):
    transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size)])
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transformations(img_lab)
    L = img_lab[[0], ...] / 50. - 1. #Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. #Between -1 and 1
    return {'L': L, 'ab': ab}

def postprocess(L, predicted_colors):
    img = lab_to_rgb(L.unsqueeze(0), predicted_colors.unsqueeze(0))[0]
    img = np.matrix.round(img*255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def predict(img, device, image_size, model):
    img = preprocess(img, image_size)
    model.to(device)
    model.eval()
    with torch.no_grad():
        predicted_colors = model(img['L'].to(device).unsqueeze(0))[0].cpu().detach()
    color_img = postprocess(img['L'].cpu(), predicted_colors)
    return color_img
           