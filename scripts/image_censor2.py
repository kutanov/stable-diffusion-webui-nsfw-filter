import torch.nn.functional as F
import torch
from PIL import Image
import open_clip
import torchvision.transforms as T
import time
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from scripts.models import NSFWModel_L14, NSFWModel_B32, NSFWModel_H14

def build_inference_model(head_path: str, model_name: str, dataset_name: str, device: str = 'cuda'):

    if model_name == 'ViT-B-32':
        head = NSFWModel_B32()
    elif model_name == 'ViT-L-14':
        head = NSFWModel_L14()
    else:
        head = NSFWModel_H14()

    head.load_state_dict(torch.load(head_path))
    head.eval()
    head.to(device)
    
    backbone = open_clip.create_model_and_transforms(model_name, dataset_name, device=device)[0].visual
    backbone.eval()

    pre_processing = T.Compose([
        T.Resize(
            size=(224, 224), 
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True),
        T.ToTensor(), 
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    return head, backbone, pre_processing

@torch.no_grad()
def predict_single_image_path(img, head, backbone, pre_processing):

    # img = Image.open(img_path)
    x = pre_processing(img).to(device)
    x = x.reshape(1, 3, 224, 224)
    p = backbone(x) 
    p = F.normalize(p)
    c = head(p)

    return c[0][0]

head, backbone, pre_procesing = build_inference_model(os.path.abspath(os.path.join(os.path.dirname(__file__), 'clip_ViT-B-32_openai_binary_nsfw_head.pth')), 
                                                      'ViT-B-32', 'openai', device)
start = time.time()

def is_image_safe(img):
    if predict_single_image_path(img, head, backbone, pre_procesing) < 0.999:
        return True
    return False
