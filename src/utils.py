import torch
from typing import List, Dict
import torchmetrics

def performance_metrics(pred, target):
    target = target.argmax(dim=1).to(torch.long)
    pred = pred.argmax(dim=1).to(torch.long)

    return {
        'accuracy': torchmetrics.functional.accuracy(pred, target).item(),
        'precision': torchmetrics.functional.precision(pred, target).item(),
        'recall': torchmetrics.functional.recall(pred, target).item(),
        'f1': torchmetrics.functional.f1_score(pred, target).item(),
    }

def avg_performance_metrics(metrics : List[Dict[str,float]]) -> Dict[str,float]:
    return {k: torch.mean(torch.stack([m[k] for m in metrics])) for k in metrics}

