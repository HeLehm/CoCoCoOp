from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from .Submodules.CLIP.clip import clip 

from tqdm import tqdm

from .utils import performance_metrics

from typing import List
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Code slightly altered from CoCoOp START
"""
def load_clip(backbone_name):
    model, preprocess = clip.load(backbone_name, device=DEVICE)
    return model, preprocess



class CachedTextEmbedder():
    def __init__(self, clip_model):
        self.model = clip_model
        self.cache = {}

    def __call__(self, text, cache=True):
        if text not in self.cache:
            with torch.no_grad():
                tok = clip.tokenize(text).to(DEVICE)
                if cache:
                    self.cache[text] = self.model.encode_text(tok)[0].cpu()
                else:
                    return self.model.encode_text(tok)[0].cpu()
        return self.cache[text]

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class FakeImageEmbeddingGenerator():
    """
    Generate fake image embeddings from random text
    """
    def __init__(self, text_encoder : CachedTextEmbedder):
        super().__init__()
        self.text_encoder = text_encoder

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
        fake_image_embeddings = []
        for text in texts:
            fake_image_embedding = self.text_encoder(text, cache=False)
            fake_image_embeddings.append(fake_image_embedding)
        return torch.stack(fake_image_embeddings)


class MetaAndScalingNet(nn.Module):
    def __init__(self, vis_dim, ctx_dim) -> None:
        super().__init__()

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.scaling_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, 1)),
            ("sigmoid", nn.Sigmoid())
        ]))
    
    def half(self):
        super().half()
        self.meta_net.half()
        self.scaling_net.half()

    def forward(self, vis_embedding):
        meta_embedding = self.meta_net(vis_embedding) # (batch, ctx_dim)
        scaling_factor = self.scaling_net(vis_embedding) # (batch, 1)
        # scale the meta embedding
        return (meta_embedding.transpose(0, 1) * scaling_factor.squeeze()).transpose(0, 1)



class PromptLearner(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx = 10,
        ctx_init = "a photo of a",
        prec = "fp32",
    ):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(DEVICE)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = ctx_vectors

        #meta and scaling net
        self.meta_scaling_net = MetaAndScalingNet(vis_dim, ctx_dim)
        
        if prec == "fp16":
            self.meta_scaling_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(DEVICE)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        return self._forward(im_features=im_features,bias_call=self.meta_scaling_net)
    
    def forward_meta_only(self, im_features):
        return self._forward(im_features=im_features,bias_call=self.meta_scaling_net.meta_net)

    def forward_scaling_only(self, im_features):
        return self.meta_scaling_net.scaling_net(im_features)
        

    def _forward(self, im_features, bias_call):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = bias_call(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts



class CustomCLIP(nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
        ctx_init = "a photo of a",
        n_ctx = 10,
        prec = "fp32",
    ):
        super().__init__()
        self.prompt_learner = PromptLearner(
                classnames=classnames,
                clip_model=clip_model,
                n_ctx=n_ctx,
                ctx_init=ctx_init,
                prec=prec,
        )
        self.clip_model = clip_model
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        #self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_features, label=None):
        return self._forward(image_features, label, self.prompt_learner.forward)

    def forward_meta_only(self, image_features, label=None):
        return self._forward(image_features, label, self.prompt_learner.forward_meta_only)

    def forward_scaling_only(self, image_features, label=None):
        scales = self.prompt_learner.forward_scaling_only(image_features)

        if self.prompt_learner.training:
            scales = scales.squeeze(1)
            loss = F.binary_cross_entropy(scales, label)
            stats = performance_metrics(scales, label, one_hot=False)
            stats['loss'] = loss.item()
            return loss, stats

        return scales

    def image_features(self, image):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features

    def _forward(self, image_features, label, prompt_call):
        tokenized_prompts = self.tokenized_prompts
       # logit_scale = self.logit_scale.exp()

        prompts = prompt_call(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = imf_i @ text_features.t() # text fwatures.t() .shape (n_cls, 512)
            logits.append(l_i)
        logits = torch.stack(logits) # (batch, n_cls)
        
        if self.prompt_learner.training:
            stats = performance_metrics(logits, label, one_hot=True)
            loss = F.cross_entropy(logits, label)
            stats['loss'] = loss.item()
            return loss,stats
        
        return logits

    
class CoCoCoOp():

    def __init__(self):
        self.cached_text_embedder = None

    def load_model(self):
        pass

    def build_model(
        self,
        classnames,
        clip_model_name,
        ctx_init = "a photo of a",
        n_ctx = 10,
        prec = "fp32",
    ):

        

        print(f"Loading CLIP model (backbone {clip_model_name})")
        clip_model, self.clip_img_preprocess = load_clip(clip_model_name)

        if prec == "fp32" or prec == "amp":
            clip_model = clip_model.float()
        self.prec = prec

        self.model = CustomCLIP(
            classnames=classnames,
            clip_model=clip_model,
            ctx_init=ctx_init,
            n_ctx=n_ctx,
            prec=prec,
        )

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        # TODO: make it possible to load model here

        self.model.to(DEVICE)

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def img_to_features(self, img):
        """
        img: PIL image
        NOTE: single image only!
        """
        img = self.clip_img_preprocess(img).unsqueeze(0).to(DEVICE)
        img_features = self.model.image_features(img)
        return img_features[0]

    def start_training(self):
        self.fake_img_emb_gen = FakeImageEmbeddingGenerator(text_encoder=self.create_cached_text_embedder())
        #TODO: schedule etc 
        self.scale_optim = torch.optim.Adam(self.model.prompt_learner.meta_scaling_net.scaling_net.parameters(), lr=1e-3)
        self.meta_optim = torch.optim.Adam(self.model.prompt_learner.meta_scaling_net.meta_net.parameters(), lr=1e-3)

    def forward_backward(self, batch):
        #TODO: amp
        self.scale_optim.zero_grad()
        self.meta_optim.zero_grad()

        image_features, label = self.parse_train_batch(batch=batch) # image: (batch, 3, 224, 224), label: (batch) (on device)
        
        s_img_features, s_labels = self.create_scaling_batch(image_features) # (batch, 512)
        s_loss, s_stats = self.model.forward_scaling_only(s_img_features, s_labels)
        s_loss.backward()
        self.scale_optim.step()

        m_loss, m_stats = self.model.forward_meta_only(image_features, label)
        m_loss.backward()
        self.meta_optim.step()

        m_stats = {f"meta_{k}": v for k, v in m_stats.items()}
        s_stats = {f"scale_{k}": v for k, v in s_stats.items()}
        stats = {**m_stats, **s_stats}

        return stats


    def create_cached_text_embedder(self):
        if self.cached_text_embedder is None:
            self.cached_text_embedder = CachedTextEmbedder(
                clip_model=self.model.clip_model
            )
        return self.cached_text_embedder


    def create_scaling_batch(self, real_img_features):
        fake_imgs = self.fake_img_emb_gen(len(real_img_features), self.model.prompt_learner.n_ctx)
        images = torch.cat([real_img_features, fake_imgs], dim=0)
        labels = torch.cat([torch.ones(len(real_img_features)), torch.zeros(len(fake_imgs))], dim=0)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        return images, labels
    
    def parse_train_batch(self, batch):
        img, label = batch
        img = img.to(DEVICE)
        label = label.to(self.model.dtype).to(DEVICE)
        return img, label

    def test(self, ds):
        
        self.model.prompt_learner.eval()
        self.model.prompt_learner.meta_scaling_net.eval()
        self.model.prompt_learner.meta_scaling_net.scaling_net.eval()
        self.model.prompt_learner.meta_scaling_net.meta_net.eval()

        self.model.prompt_learner.meta_scaling_net.scaling_net.eval()
        self.model.prompt_learner.meta_scaling_net.meta_net.eval()

        n = 0
        stats_avg = {}
        for batch in tqdm(ds, desc="Testing"):
            image_features, label = self.parse_train_batch(batch=batch)
            logits = self.model.forward(image_features, label)
            stats = performance_metrics(logits, label)

            for k, v in stats.items():
                if k not in stats_avg:
                    stats_avg[k] = []
                stats_avg[k].append(v)
            n += 1
        return {k: sum(v) / n for k, v in stats_avg.items()}

