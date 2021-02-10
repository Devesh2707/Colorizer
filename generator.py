import torch
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
import config

weights_path = config.MODEL_WEIGHTS

def build_res_unet(n_input=1, n_output=2, size=224):
    body = create_body(resnet18, pretrained=False, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

def load_weights(device, image_size):
    net_G = build_res_unet(size = image_size)
    net_G.load_state_dict(torch.load(weights_path, map_location=device))
    return net_G