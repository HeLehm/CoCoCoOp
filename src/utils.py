import torch
from typing import List, Dict
import torchmetrics

def performance_metrics(pred, target, one_hot=True):
    if one_hot:
        target = target.argmax(dim=1).to(torch.long)
        pred = pred.argmax(dim=1).to(torch.long)
    else:
        target = target.to(torch.long)
        pred = pred.to(torch.long)
    
    return {
        'accuracy': torchmetrics.functional.accuracy(pred, target).item(),
    }

def avg_performance_metrics(metrics : List[Dict[str,float]]) -> Dict[str,float]:
    return {
        k: sum([m[k] for m in metrics]) / len(metrics) for k in metrics[0].keys()
    }

