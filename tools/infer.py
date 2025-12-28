import argparse
import os
import yaml
import sys
import cv2
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.pipeline.engine import InferenceEngine

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
    parser = argparse.ArgumentParser(description="Image Retargeting Inference")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to config file")
    parser.add_argument("--image_dir", type=str, help="Override input image directory")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.image_dir: cfg.inference.input_dir = args.image_dir
    if args.output_dir: cfg.inference.output_dir = args.output_dir

    engine = InferenceEngine(cfg)
    
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(cfg.inference.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    print(f"Found {len(image_files)} images. Starting inference...")

    for img_file in tqdm(image_files):
        image_path = os.path.join(cfg.inference.input_dir, img_file)
    
        lama_path = os.path.join(cfg.inference.llama_dir, img_file)
        
        if not os.path.exists(lama_path):
            lama_path = image_path

        for ratio in cfg.inference.target_ratios:
            try:
                result = engine.run(image_path, lama_path, target_ratio=ratio)
                
                if result is not None:
                    ratio_str = f"{ratio:.2f}".replace('.', '_')
                    save_dir = os.path.join(cfg.inference.output_dir, f"ratio_{ratio_str}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    cv2.imwrite(os.path.join(save_dir, img_file), result)
            
            except Exception as e:
                print(f"Error processing {img_file} at ratio {ratio}: {e}")


if __name__ == "__main__":
    main()