from .models import CoCoCoOp
from .utils import load_clip

from collections import OrderedDict

import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from .Submodules.CLIP.clip import clip
from .Submodules.torch_warmup_lr.torch_warmup_lr.wrappers import WarmupLR 

from tqdm import tqdm

from .utils import performance_metrics, avg_performance_metrics

from typing import List
import random

from .models import TextEncoder

class FakeImageEmbeddingGenerator():
    """
    Generate fake image embeddings from random text
    """
    def __init__(self, text_encoder : TextEncoder, device : torch.device):
        super().__init__()
        self.text_encoder = text_encoder
        self.device = device

    def random_text(self, batch_size : int, n_ctx : int) -> List[str]:
        # generate random text
        text = []
        #TODO: make sure no special tokens are used
        vocab = list(clip._tokenizer.encoder.keys())
        for _ in range(batch_size):
            text.append("".join([random.choice(vocab) for _ in range(n_ctx)]))
        return text
        
    def __call__(self, batch_size, n_ctx):
        texts = self.random_text(batch_size, n_ctx)
        tokenized_texts = clip.tokenize(texts).to(self.device)
        return self.text_encoder(tokenized_texts, None)

class CoCoCoOpTrainer():

    def build_model(
        self,
        config,
        device,
    ):
        self.device = device

        clip_model, clip_preprocess = load_clip(config.clip_backbone)

        self.model = CoCoCoOp(
            clip_model,
            clip_preprocess,
            ctx_init = config.ctx_init,
            device=self.device,
        )

        self.fake_img_emb_gen = FakeImageEmbeddingGenerator(self.model.text_encoder, self.device)

        self.model = self.model.to(self.device)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "net"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

         # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.fake_img_emb_gen = FakeImageEmbeddingGenerator(text_encoder=self.model.text_encoder, device=self.device)


    def forward_step(self, batch, lwf_beta=0.1):

        img_features, labels = self.parse_train_batch(batch)
        s_images, s_labels = self.create_scaling_batch(img_features)

 
        meta_logits = self.model.forward_meta_only(img_features)
        scale_logits = self.model.forward_scale_only(s_images)

        meta_ce_loss = F.cross_entropy(meta_logits, labels)
        scale_ce_loss = F.binary_cross_entropy(scale_logits, s_labels)

        meta_loss = meta_ce_loss
        scale_loss = scale_ce_loss

        meta_lwf_loss = None
        scale_lwf_loss = None

        if lwf_beta is not None:
            assert self.prev_model is not None, "Previous model is not defined"

            old_meta_logits = self.prev_model.forward_meta_only(img_features)
            old_scale_logits = self.prev_model.forward_scale_only(s_images)

            n_classes = len(self.prev_model.classnames)
            old_classes_meta_logits = meta_logits[:, :n_classes]
            old_classes_scale_logits = scale_logits[:, :n_classes]

            meta_lwf_loss = F.cross_entropy(old_classes_meta_logits, old_meta_logits) * self.lwf_beta
            scale_lwf_loss = F.binary_cross_entropy(old_classes_scale_logits, old_scale_logits) * self.lwf_beta

            meta_loss += meta_lwf_loss
            scale_loss += scale_lwf_loss

        meta_stats = performance_metrics(meta_logits, labels, one_hot=True)
        meta_stats["ce_loss"] = meta_ce_loss.item()
        meta_stats["loss"] = meta_loss.item()
        if meta_lwf_loss is not None:
            meta_stats["lwf_loss"] = meta_lwf_loss.item()

        scale_stats = performance_metrics(scale_logits, s_labels, one_hot=False)
        scale_stats["bce_loss"] = scale_ce_loss.item()
        scale_stats["loss"] = scale_loss.item()
        if scale_lwf_loss is not None:
            scale_stats["lwf_loss"] = scale_lwf_loss.item()

        meta_stats = {'meta_' + k: v for k, v in meta_stats.items()}
        scale_stats = {'scale_' + k: v for k, v in scale_stats.items()}

        stats = {**meta_stats, **scale_stats}

        return meta_loss, scale_loss, stats

    def train_step(self, batch, lwf_beta = 0.1):
        self.model.train()
        #optimizer
        self.scale_optim.zero_grad()
        self.meta_optim.zero_grad()

        #forward step with stats
        meta_loss, scale_loss, stats = self.forward_step(batch, lwf_beta)

        #backward step
        meta_loss.backward()
        scale_loss.backward()
        self.scale_optim.step()
        self.meta_optim.step()

        return stats

    def test(self, dataloader):
        self.model.eval()
        stats = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                _, _, batch_stats = self.forward_step(batch, None)
                stats.append(batch_stats)
        stats = avg_performance_metrics(stats)
        return stats
    
    def schedule_step(self):
        self.scale_scheduler.step()
        self.meta_scheduler.step()

    def init_opt(self, config):
        optimizer = config.optimizer
        scheduler = config.lr_scheduler

        self.scale_optim = optimizer(self.model.prompt_learner.scaling_net.parameters(), lr=config.lr)
        self.meta_optim = optimizer(self.model.prompt_learner.meta_net.parameters(), lr=config.lr)

        self.scale_scheduler = scheduler(self.scale_optim, **config.lr_scheduler_kwargs)
        self.scale_scheduler = WarmupLR(self.scale_scheduler, **config.lr_scheduler_warmup_kwargs)

        self.meta_scheduler = scheduler(self.meta_optim, **config.lr_scheduler_kwargs)
        self.meta_scheduler = WarmupLR(self.meta_scheduler, **config.lr_scheduler_warmup_kwargs)

    def create_scaling_batch(self, real_img_features):
        fake_imgs = self.fake_img_emb_gen(len(real_img_features), self.model.prompt_learner.n_ctx).to(self.device)
        images = torch.cat([real_img_features, fake_imgs], dim=0)
        labels = torch.cat([torch.ones(len(real_img_features)), torch.zeros(len(fake_imgs))], dim=0)
        labels = labels.to(self.device)
        return images, labels

    def parse_train_batch(self, batch):
        img, label = batch
        img = img.to(self.device)
        label = label.to(self.model.dtype).to(self.device)
        return img, label


