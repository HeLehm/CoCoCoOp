import torch
from typing import List, Dict
import torchmetrics
from .Submodules.CLIP.clip import clip

def load_clip(backbone_name, device):
    model, preprocess = clip.load(backbone_name, device=device)
    return model, preprocess

def clip_encode_text(text, clip_model):
    return clip_model.encode_text(clip.tokenize(text).to(clip_model.device))

def performance_metrics(pred, target, one_hot=True):
    if one_hot:
        target = target.argmax(dim=1).to(torch.long)
        pred = pred.argmax(dim=1).to(torch.long)
    else:
        assert target.ndim == 1 and target.size() == pred.size()
        pred = pred > 0.5
        return {
            'accuracy' : (target == pred).sum().item() / target.size(0)
        }
    
    return {
        'accuracy': torchmetrics.functional.accuracy(pred, target).item(),
    }

def avg_performance_metrics(metrics : List[Dict[str,float]]) -> Dict[str,float]:
    return {
        k: sum([m[k] for m in metrics]) / len(metrics) for k in metrics[0].keys()
    }

def frange(start, stop, step, skip_first=True):
    """
    A range function, that does accept float increments
    """
    i = start
    is_first_val = True
    while i < stop:
        if not is_first_val or not skip_first:
            yield i
        is_first_val = False
        i += step
