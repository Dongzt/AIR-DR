import torch
import cv2
import numpy as np
import os
from diffusers.utils import load_image
from src.core.models import InstanceRelocationModel_Tiny, InstanceRelocationModel
from src.core.retarget import filter_bounding_boxes, generate_layout, add_bg, create_inpaint_map, single_process_function
from src.core.dataset import get_transform_original 
from src.utils.image_utils import compute_target_size

from .segmentor import Segmentor
from .inpainter import FluxInpainter

class InferenceEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.models.device)
        self.base_ratio = [16/9, 1.0, 4/3, 9/16]
        self._init_models()
        
        self.transform = get_transform_original(target_size=(cfg.models.layout.input_size, cfg.models.layout.input_size))

    def _init_models(self):
        # MMDetection
        self.segmentor = Segmentor(
            self.cfg.models.mmdet.config, 
            self.cfg.models.mmdet.checkpoint, 
            self.device
        )
        
        # Layout Model
        model_cls = InstanceRelocationModel_Tiny if self.cfg.models.layout.type == 'tiny' else InstanceRelocationModel
        self.layout_model = model_cls(num_ratios=4, img_size=self.cfg.models.layout.input_size)
        self.layout_model.load_state_dict(torch.load(self.cfg.models.layout.checkpoint, map_location=self.device))
        self.layout_model.to(self.device).eval()
        
        # Inpainter
        self.inpainter = FluxInpainter(
            self.cfg.models.inpainting.fill_model,
            self.cfg.models.inpainting.redux_model,
            self.device,
            token=self.cfg.inference.hf_token
        )

    def predict_offsets(self, mask):
        # mask: numpy (H, W) -> tensor (1, 1, 640, 640) via transform
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        input_tensor = self.transform(mask_tensor).unsqueeze(0).to(self.device) # (1, 1, 640, 640)
        
        with torch.no_grad():
            preds = self.layout_model(input_tensor) # (1, 32)
        
        offsets_flat = preds.squeeze(0).cpu().numpy()
        offsets_list = np.split(offsets_flat, 4) 
        
        return offsets_list

    def run(self, image_path, lama_path, target_ratio):

        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Image not found: {image_path}")
        
        mask = self.segmentor.predict(image_path)
        
        instances = filter_bounding_boxes(mask, self.cfg.inference.min_instance_area)
        if len(instances) < 1 or len(instances) > 8:
            return None

        all_offsets = self.predict_offsets(mask)
        print(len(all_offsets))
        print(all_offsets)

        best_idx = np.argmin([abs(target_ratio - r) for r in self.base_ratio])
        offset_x = all_offsets[best_idx]

        if len(instances) == 1:
             paint_bg, _, paint_flag = single_process_function(image, mask, target_ratio, instances[0].bbox)
             if not paint_flag:
                 return paint_bg

        retarget_mask, target_x, target_y, if_resize = generate_layout(mask, target_ratio, instances, offset_x)
        
        if retarget_mask is None: return None

        paint_bg = create_inpaint_map(image, mask, retarget_mask, target_x, target_y)
        new_paint_bg, new_retarget_mask = add_bg(mask, image, retarget_mask, paint_bg)
        
        inverted_mask = np.where(new_retarget_mask == 0, 255, 0).astype(np.uint8)

        layout_h, layout_w = new_paint_bg.shape[:2]

        gen_h, gen_w = compute_target_size(target_ratio)
        
        input_bg_resized = cv2.resize(new_paint_bg, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)
        input_mask_resized = cv2.resize(inverted_mask, (gen_w, gen_h), interpolation=cv2.INTER_NEAREST)
        
        ref_img = load_image(lama_path).resize((gen_w, gen_h))

        generated_image = self.inpainter.predict(
            input_bg_resized, 
            input_mask_resized, 
            ref_img, 
            height=gen_h, 
            width=gen_w,
            seed=self.cfg.models.inpainting.seed,
            steps=self.cfg.models.inpainting.num_steps,
            guidance=self.cfg.models.inpainting.guidance_scale
        )
        
        final_image = cv2.resize(generated_image, (layout_w, layout_h), interpolation=cv2.INTER_AREA)

        return final_image