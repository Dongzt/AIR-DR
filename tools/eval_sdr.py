import argparse
import os
import sys
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.getcwd())

from src.pipeline.segmentor import Segmentor
from src.core.metric import compute_sdr_score
class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"No such attribute: {name}")

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return ConfigDict(cfg)

def main():
    parser = argparse.ArgumentParser(description="Calculate SDR Metric with On-the-fly Segmentation")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to config file")
    parser.add_argument("--ori_image_dir", type=str, required=True, help="Directory containing ORIGINAL RGB images")
    parser.add_argument("--retarget_dir", type=str, required=True, help="Root directory of RETARGETED results (can contain subdirs)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Initializing Segmentor...")
    segmentor = Segmentor(
        config_path=cfg.models.mmdet.config,
        checkpoint_path=cfg.models.mmdet.checkpoint,
        device=cfg.models.device
    )

    ori_mask_cache = {}

    results = defaultdict(list)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    print(f"Start Evaluation recursively on: {args.retarget_dir}")
    print("-" * 60)

    for root, dirs, files in os.walk(args.retarget_dir):
        
        img_files = [f for f in files if f.lower().endswith(image_extensions)]
        
        if not img_files:
            continue
            
        print(f"Processing directory: {os.path.relpath(root, args.retarget_dir)}")
        
        dir_sdr_list = []
        
        for filename in tqdm(img_files, leave=False):
            
            retarget_path = os.path.join(root, filename)
            ori_path = os.path.join(args.ori_image_dir, filename)

            if not os.path.exists(ori_path):
            
                found = False
                for ext in image_extensions:
                    temp_path = os.path.splitext(ori_path)[0] + ext
                    if os.path.exists(temp_path):
                        ori_path = temp_path
                        found = True
                        break
                if not found:
                    continue 
            
            if ori_path in ori_mask_cache:
                mask_ori = ori_mask_cache[ori_path]
            else:
                try:
                    mask_ori = segmentor.predict(ori_path)
                    ori_mask_cache[ori_path] = mask_ori
                except Exception as e:
                    print(f"Error segmenting original {filename}: {e}")
                    continue

            try:
                mask_retarget = segmentor.predict(retarget_path)
            except Exception as e:
                print(f"Error segmenting result {filename}: {e}")
                continue

            sdr = compute_sdr_score(mask_ori, mask_retarget)
            
            if sdr is not None:
                dir_sdr_list.append(sdr)

        if dir_sdr_list:
            avg_sdr = np.mean(dir_sdr_list)
            results[root].append(avg_sdr)
            print(f"  -> Avg SDR: {avg_sdr:.4f} (Count: {len(dir_sdr_list)})")
        else:
            print(f"  -> No valid SDR calculated.")

    print("-" * 60)
    print("Final Summary:")
    
    total_sdr = []
    for dir_path, scores in results.items():
        rel_path = os.path.relpath(dir_path, args.retarget_dir)
        val = scores[0]
        total_sdr.append(val)
        print(f"{rel_path:<40} : {val:.4f}")

    if total_sdr:
        print(f"\nOverall Average SDR: {np.mean(total_sdr):.4f}")

if __name__ == "__main__":
    main()