import torch
import numpy as np
from mmdet.apis import DetInferencer

class Segmentor:
    def __init__(self, config_path, checkpoint_path, device):
        self.inferencer = DetInferencer(
            model=config_path,
            weights=checkpoint_path,
            device=device
        )

    def predict(self, image_path):
        
        results = self.inferencer(image_path, return_datasamples=True, no_save_vis=True)
        
        masks = results['predictions'][0].pred_instances.masks
        
        h, w = masks[0].shape[-2:] 
        binary_mask = np.zeros((h, w), dtype=np.uint8)

        for mask in masks:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            mask_area = np.sum(mask_np > 0)
            overlap_area = np.sum(binary_mask[mask_np > 0] > 0)
            
            if mask_area > 0:
                overlap_ratio = overlap_area / mask_area
            else:
                overlap_ratio = 0
            
            if overlap_ratio > 0.9:
                continue

            binary_mask[mask_np > 0] = 255

        return binary_mask