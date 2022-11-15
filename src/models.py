from collections import OrderedDict

import torch
import torch.nn as nn

from .Submodules.CLIP.clip import clip


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts=None):
        if tokenized_prompts is None:
            x = self.token_embedding(prompts).type(self.dtype)
            x = x + self.positional_embedding.type(self.dtype)
        else:
            x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if tokenized_prompts is None:
            x = x[torch.arange(x.shape[0]), prompts.argmax(dim=-1)] @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        device,
        ctx_init = "a photo of a",
    ) -> None:
        super().__init__()

        self.clip_model = clip_model

        self.dtype = clip_model.dtype
        self.device = device
    
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        # init all context values (besides the classes)
        self._init_ctx(ctx_init=ctx_init)
        # init the two nets (sclaing and meta)
        self._init_nets(
            vis_dim=vis_dim,
            ctx_dim=ctx_dim,
        )
        

    def forward(self, im_features):
        return self._forward(
            im_features=im_features,
            bias_call=self._forward_both_nets,
        )

    def forward_meta_only(self, im_features):
        return self._forward(
            im_features=im_features,
            bias_call=self.meta_net,
        )

    def _forward_both_nets(self, vis_embedding):
        meta_embedding = self.meta_net(vis_embedding) # (batch, ctx_dim)
        scaling_factor = self.scaling_net(vis_embedding) # (batch, 1)
        # scale the meta embedding
        return (meta_embedding.transpose(0, 1) * scaling_factor.squeeze()).transpose(0, 1)
    
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

    def _init_nets(self, vis_dim, ctx_dim):
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

    def _init_ctx(self, ctx_init):
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]

        self.prompt_prefix = ctx_init
        self.ctx = ctx_vectors.to(self.device)
        self.n_ctx = n_ctx

    def set_class_names(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]

        self.classnames = classnames

        #CoCoOp Code
        prompts = [self.prompt_prefix + " " + name + "." for name in self.classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS

        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor


    def add_class_names(self, classnames):
        # if these are the first classes, init the class names
        if not hasattr(self,'classnames'):
            self.classnames = []

        # add new class names
        self.classnames = self.classnames +  classnames 

        self.set_class_names(self.classnames)

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

    
class CoCoCoOp(nn.Module):
    def __init__(
        self,
        clip_model,
        clip_preprocess,
        device,
        ctx_init = "a photo of a",
    ) -> None:
        super().__init__()

        self.preprocess = clip_preprocess

        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        self.prompt_learner = PromptLearner(clip_model, device, ctx_init).to(device).to(self.dtype)
        

        self.text_encoder = TextEncoder(clip_model).to(self.dtype)
        self.text_encoder.training = False


    """
    classes realted functions
    """
    @property
    def classnames(self):
        return self.prompt_learner.classnames

    def add_class_names(self, classnames):
        self.prompt_learner.add_class_names(classnames)

    def set_class_names(self, classnames):
        self.prompt_learner.set_class_names(classnames)
    
    """
    loading/saving
    """
    def load(self, path):
        pass

    def save(self, path):
        pass
    
    """
    Forward passes
    """

    def get_image_features(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.prompt_learner.device)
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features[0]

    def forward(self, image_features):
        return self._forward(image_features, self.prompt_learner.forward)

    def forward_meta_only(self, image_features):
        return self._forward(image_features, self.prompt_learner.forward_meta_only)

    def forward_scale_only(self, image_features):
        return self.prompt_learner.scaling_net(image_features).squeeze(-1)

    """
    PRIVATE Functions
    """
    def _forward(self, image_features, prompt_call):

        tokenized_prompts = self.prompt_learner.tokenized_prompts
       # logit_scale = self.logit_scale.exp()

        prompts = prompt_call(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = imf_i @ text_features.t() # text fwatures.t() .shape (n_cls, 512)
            logits.append(l_i)
        logits = torch.stack(logits) # (batch, n_cls)
        
        return logits
