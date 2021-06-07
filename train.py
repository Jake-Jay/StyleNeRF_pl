import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from opt import get_opts
import torch
import torchvision.transforms as transforms
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict, FeatureLoss

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger

def image_loader(image_name, imsize):
    image = Image.open(image_name)
    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
         transforms.ToTensor()]     # transform it into a torch tensor
    )  

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.stage = hparams.stage

        if self.stage == 'style':
            style_img_path = self.hparams.style_img
            print(style_img_path)
            self.style_img = image_loader(
                image_name=style_img_path,
                imsize=self.hparams.img_wh[0]
            )
            self.loss = FeatureLoss(
                style_img=self.style_img,
                style_weight=1000000,
                content_weight=1
            )
        else:
            self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF(stage=self.stage)
        if self.stage == 'style':
            load_ckpt(self.nerf_coarse, hparams.ckpt_path, model_name='nerf_coarse')
        self.models = [self.nerf_coarse]
        
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(stage=self.stage)
            if self.stage == 'style':
                load_ckpt(self.nerf_fine, hparams.ckpt_path, model_name='nerf_fine')
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        # TODO you can also collect the valid mask here during val/style
        
        # validation step (stage = style):
        # - rays.shape = [1, w*h, 8]
        # - rgbs.shape =[1, w*h, 3]

        # train step (stage = style):
        # - rays.shape = [1, w*h, 8]
        # - rgbs.shape =[1, w*h, 3]

        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        
        self.train_dataset = dataset(
            split='train', 
            stage=self.stage,
            **kwargs
        )
        self.val_dataset = dataset(
            split='val',
            stage=self.stage,
            **kwargs
        )

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        if self.stage == 'density':
            return DataLoader(
                self.train_dataset,
                shuffle=True,
                num_workers=4,
                batch_size=self.hparams.batch_size,
                pin_memory=True
            )
        elif self.stage == 'style':
            return DataLoader(
                self.train_dataset,
                shuffle=True,
                num_workers=4,
                batch_size=1, # style one image at a time
                pin_memory=True
            )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=1, # set back to 4 after debug
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        if self.stage == 'density':
            rays, rgbs = self.decode_batch(batch)
            results = self(rays)
            log['train/loss'] = loss = self.loss(results, rgbs)
        
        elif self.stage == 'style':
            rays, rgbs = self.decode_batch(batch)
            rays = rays.squeeze() # (H*W, 8)
            rgbs = rgbs.squeeze() # (H*W, 3)
            results = self(rays)
            target = self._prepare_for_feature_loss(rgbs) #(1,3,W,H)

            course_result = self._prepare_for_feature_loss(results['rgb_coarse']) #(1,3,W,H)
            course_loss, _, _ = self.loss(course_result, target)

            if 'rgb_fine' in results:
                fine_result = self._prepare_for_feature_loss(results['rgb_fine']) #(1,3,W,H)
                fine_loss, _, _  = self.loss(fine_result, target)
                loss = fine_loss + course_loss
            else:
                loss = course_loss

            log['train/loss'] = loss

        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }
    
    def _prepare_for_feature_loss(self, img:torch.tensor):
        '''img of shape (H*W, 3) -> (1, 3, w, h)'''
        img = img.permute(1,0) #(3, H*W)
        img = img.view(3, self.hparams.img_wh[0], self.hparams.img_wh[1]) #(3,W,H)
        img = img.unsqueeze(0) # (1,3,W,H)
        return img

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 8)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)

        if self.stage == 'density':
            log = {'val_loss': self.loss(results, rgbs)}
        
        elif self.stage == 'style':
            target = self._prepare_for_feature_loss(rgbs) #(1,3,W,H)

            course_result = self._prepare_for_feature_loss(results['rgb_coarse']) #(1,3,W,H)
            course_loss, _, _ = self.loss(course_result, target)

            if 'rgb_fine' in results:
                fine_result = self._prepare_for_feature_loss(results['rgb_fine']) #(1,3,W,H)
                fine_loss, _, _  = self.loss(fine_result, target)
                loss = fine_loss + course_loss
            else:
                loss = course_loss

            log = {'val_loss': loss}

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(f'ckpts/{hparams.exp_name}','{epoch:d}'),
        monitor='val/loss',
        mode='min',
        save_top_k=5,)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                    #   resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                    #   early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend= None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1)

    trainer.fit(system)

