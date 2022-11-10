import torch
from typing import List, Dict
import torchmetrics

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

