import os
import argparse
import yaml
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import sys
sys.path.append(os.getcwd()) 

from src.core.models import InstanceRelocationModel_Tiny, InstanceRelocationModel
from src.core.dataset import build_dataloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.system.device)
        self.output_dir = cfg.train.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.train_loader = build_dataloader(cfg, split='train')
        self.val_loader = build_dataloader(cfg, split='test')
        
        model_cls = InstanceRelocationModel_Tiny if cfg.model.type == 'tiny' else InstanceRelocationModel
        self.model = model_cls(
            num_ratios=len(cfg.dataset.ratios),
            img_size=cfg.dataset.img_size[0]
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=cfg.train.lr, 
            weight_decay=cfg.train.weight_decay
        )

        warmup_epochs = cfg.train.scheduler.warmup_epochs
        total_epochs = cfg.train.epochs
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs),
                CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=cfg.train.scheduler.min_lr)
            ],
            milestones=[warmup_epochs]
        )
        
        self.criterion = nn.MSELoss() if cfg.train.loss == 'mse' else nn.SmoothL1Loss(beta=1.0)

        self.best_val_loss = float('inf')

    def calculate_loss(self, pred_all, target_all):

        num_ratios = len(self.cfg.dataset.ratios)
        pred_reshaped = pred_all.view(-1, num_ratios, 8)
        target_reshaped = target_all.view(-1, num_ratios, 8)
        
        batch_loss = 0.0
        batch_size = pred_all.size(0)
        
        for i in range(batch_size):
            sample_loss = 0.0
            for ratio_idx in range(num_ratios):
                pred = pred_reshaped[i, ratio_idx]
                true = target_reshaped[i, ratio_idx]
                
                valid_mask = (true != 300)
                n = valid_mask.sum().item()
                
                if n > 0:
                    ratio_loss = self.criterion(pred[:n], true[:n])
                    sample_loss += ratio_loss
            
            batch_loss += sample_loss
            
        return batch_loss / batch_size

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.train.epochs}")
        for input_masks, targets in pbar:
            input_masks = input_masks.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(input_masks)
            
            loss = self.calculate_loss(preds, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        self.scheduler.step()
        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for input_masks, targets in self.val_loader:
                input_masks = input_masks.to(self.device)
                targets = targets.to(self.device)
                
                preds = self.model(input_masks)
                loss = self.calculate_loss(preds, targets)
                val_loss += loss.item()
                
        return val_loss / len(self.val_loader)

    def run(self):
        logger.info(f"Start training on {self.device}...")
        
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
            
            if epoch % self.cfg.train.val_interval == 0 or epoch == 1:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, "best.pth")
                
                self._save_checkpoint(epoch, f"epoch_{epoch}.pth")

    def _save_checkpoint(self, epoch, name):
        path = os.path.join(self.output_dir, name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg, 
        }, path)
        logger.info(f"Saved model to {path}")

class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"No such attribute: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_layout.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = ConfigDict(cfg_dict)
    
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == '__main__':
    main()