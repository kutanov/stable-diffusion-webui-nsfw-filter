import os.path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers.utils import logging
from torchvision.utils import save_image
import torchvision.transforms as T
transformTorchToPil = T.ToPILImage()


# from scripts.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor

# fast nsfw
from scripts.image_censor import model as onnx_model
from scripts.prompt_censor import is_prompt_safe

from modules import scripts, images


logger = logging.get_logger(__name__)

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None

warning_image = os.path.join("extensions", "stable-diffusion-webui-nsfw-filter", "warning", "warning.png")

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def censor_batch(x, safety_checker_adj: float):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    pil_images = numpy_to_pil(x_samples_ddim_numpy)
    predictions = [onnx_model.predict(x_sample) for x_sample in pil_images]
    
    index = 0
    for p in predictions:
        try:
            safety = next(predicate for predicate in p['predictions'] if predicate['label'] == 'Safe')['confidence']
            naked = next(predicate for predicate in p['predictions'] if predicate['label'] == 'Naked')['confidence']
            if safety < 0.7 or naked > 0.5: 
                hwc = x.shape
                y = Image.open(warning_image).convert("RGB").resize((hwc[3], hwc[2]))
                y = (np.array(y) / 255.0).astype("float32")
                y = torch.from_numpy(y)
                y = torch.unsqueeze(y, 0).permute(0, 3, 1, 2)
                x[index] = y
            index += 1
        except Exception as e:
            logger.warning(e)
            index += 1

    return x

# def censor_batch(x, safety_checker_adj: float):
#     pil_images = numpy_to_pil(x)
#     predictions = [onnx_model.predict(x_sample) for x_sample in pil_images]
#     print(predictions)
#     x = torch.from_numpy(x).permute(0, 3, 1, 2)
    
#     index = 0
#     for p in predictions:
#         try:
#             safety = next(predicate for predicate in p['predictions'] if predicate['label'] == 'Safe')['confidence']
#             naked = next(predicate for predicate in p['predictions'] if predicate['label'] == 'Naked')['confidence']
#             if safety < 0.7 or naked > 0.5: 
#                 hwc = x.shape
#                 y = Image.open(warning_image).convert("RGB").resize((hwc[3], hwc[2]))
#                 y = (np.array(y) / 255.0).astype("float32")
#                 y = torch.from_numpy(y)
#                 y = torch.unsqueeze(y, 0).permute(0, 3, 1, 2)
#                 try:
#                     images.save_image(transformTorchToPil(x[index].permute(0, 2, 3, 1)), p.outpath_samples, "", forced_filename=f"before_nsfw")
#                 except Exception:
#                     print(f"ERROR saving generated image to path: {p.outpath_samples}")
#                 x[index] = y
#             index += 1
#         except Exception as e:
#             logger.warning(e)
#             print(e)
#             index += 1

#     return x.permute(0, 2, 3, 1)


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    # def process(self, p, *args):
    #     if is_prompt_safe(p.prompt) is False:
    #         print("prompt is unsafe " + p.prompt)
    #         p.prompt = ''
        

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Args:
            p:
            *args:
                args[0]: enable_nsfw_filer. True: NSFW filter enabled; False: NSFW filter disabled
                args[1]: safety_checker_adj
            **kwargs:
        Returns:
            images
        """
        images = kwargs['images']
        x = torch.from_numpy(images).permute(0, 3, 1, 2)
        if args[0] is True:
            if is_prompt_safe(p.prompt) is False:
                print('unsafe prompt' + p.prompt)
                index = 0
                for image in images:
                    hwc = x.shape
                    y = Image.open(warning_image).convert("RGB").resize((hwc[3], hwc[2]))
                    y = (np.array(y) / 255.0).astype("float32")
                    y = torch.from_numpy(y)
                    y = torch.unsqueeze(y, 0).permute(0, 3, 1, 2)
                    try:
                        images.save_image(transformTorchToPil(x[index].permute(0, 2, 3, 1)), p.outpath_samples, "", forced_filename=f"before_nsfw")
                    except Exception:
                        print(f"ERROR saving generated image to path: {p.outpath_samples}")
                    x[index] = y
                images[:] = x.permute(0, 2, 3, 1)
            else:
                images[:] = censor_batch(images, args[1])[:]

    def ui(self, is_img2img):
        enable_nsfw_filer = gr.Checkbox(label='Enable NSFW filter',
                                        value=False,
                                        elem_id=self.elem_id("enable_nsfw_filer"))
        safety_checker_adj = gr.Slider(label="Safety checker adjustment",
                                       minimum=-0.5, maximum=0.5, value=0.0, step=0.001,
                                       elem_id=self.elem_id("safety_checker_adj"))
        return [enable_nsfw_filer, safety_checker_adj]
