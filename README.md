<div align="center">
<h1>AIR-DR (AAAI2026)</h1>
<h2>Adaptive Image Retargeting with Instance Relocation and Dual-guidance Repainting
</h2>
</div>

## Introduction
AIR-DR is an image retargeting method with foreground instance relocation and background guidance repainting. AIR-DR supports retargeting of any target ratio.

<p align="center">
  <img src="./assets/AIR-DR.png" width="800" />
</p>

## Inference

Running our method requires obtaining [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), [LaMa](https://github.com/advimman/lama), and [FLUX.1 Fill & Redux](https://github.com/black-forest-labs/flux). Thanks for these excellent works.

Get [weights](https://pan.baidu.com/s/1vl4edU3uTP2pLtQxoeV_1Q?pwd=wsgp) and move them to ./weights

Using the following command to inference:
```bash
python tools/infer.py --config configs/inference.yaml
```

## Training and Evaluation
```bash
# Training
python tools/train_net.py --config configs/train_layout.yaml

# Evaluation
python tools/eval_sdr.py \
  --config configs/inference.yaml \
  --ori_image_dir your_input_path \
  --retarget_dir your_results_path
```

