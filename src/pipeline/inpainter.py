import torch
import cv2
import numpy as np
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from src.utils.image_utils import numpy_to_pil

class FluxInpainter:
    def __init__(self, fill_model_id, redux_model_id, device, token=None):
        self.device = device
        self.token = token
        
        self.pipe_redux = FluxPriorReduxPipeline.from_pretrained(
            redux_model_id, 
            torch_dtype=torch.bfloat16, 
            token=token
        ).to(device)
        
        self.pipe_fill = FluxFillPipeline.from_pretrained(
            fill_model_id, 
            torch_dtype=torch.bfloat16, 
            token=token,
            text_encoder=None,
            text_encoder_2=None
        ).to(device)

    def predict(self, bg_image, mask_image, ref_image, height, width, seed=0, steps=10, guidance=30):

        img_pil = numpy_to_pil(bg_image)
        mask_pil = numpy_to_pil(mask_image, mode="L")
        
        pipe_prior_output = self.pipe_redux(ref_image)

        generator = torch.Generator(self.device).manual_seed(seed)
        result = self.pipe_fill(
            image=img_pil,
            mask_image=mask_pil,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=512,
            generator=generator,
            **pipe_prior_output,
            output_type="np"
        ).images[0]
        
        # (RGB -> BGR, 0-1 -> 0-255)
        result_uint8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
        
        return result_bgr