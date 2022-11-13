from src.trainers import CoCoCoOpTrainer
from src.datasets import get_dataset_class, DatasetSampler
from src.utils import avg_performance_metrics, frange
import wandb
from tqdm import tqdm
import torch
import copy

from src.utils import frange

from types import SimpleNamespace

this_config = {
    'CL_config' : SimpleNamespace(**{
        'type' : 'class_incremental',
        'steps' : 10,
        'lwf_beta' : 0.5
    }),
    'Training_ds': 'Caltech101',
    'batch_size': 1,
    'epochs': 10,
    'lr': 0.002,
    'optimizer': torch.optim.SGD,
    'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR,
    'lr_scheduler_kwargs': {'T_max': 10},
    'lr_scheduler_warmup_kwargs': {'init_lr':0.01, 'warmup_strategy': 'constant', 'num_warmup': 1},
    'clip_backbone': 'ViT-B/16',
    'ctx_init': 'a photo of a',
}

def run(config, with_wandb=True):
    config = SimpleNamespace(**config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = CoCoCoOpTrainer()
    trainer.build_model(config, device=device)

    
    last_f_val = 0.
    for i,f in enumerate(frange(0., 1., 1./config.CL_config.steps)):

        lwf_beta = config.CL_config.lwf_beta if i > 0 and hasattr(config, 'CL_config') and hasattr(config.CL_config, 'lwf_beta') else None
        
        dataset_kwargs = {
            'one_hot_encode_labels' : True,
            'download' : True,
        }

        ds_class = get_dataset_class(config.Training_ds)

        train_ds = ds_class(split='train', class_slices=(last_f_val, f), **dataset_kwargs)
        val_ds = ds_class(split='val', class_slices=(last_f_val, f), **dataset_kwargs)
        val_ds_full = ds_class(split='val',class_slices=(0., f), **dataset_kwargs)
        last_f_val = f

        train_sampler = DatasetSampler(train_ds, batch_size=config.batch_size)
        val_sampler = DatasetSampler(val_ds, batch_size=config.batch_size)
        val_sampler_full = DatasetSampler(val_ds_full, batch_size=config.batch_size)

        trainer.model.add_class_names(train_ds.get_active_class_names())

        train_ds.transform = lambda x: trainer.model.get_image_features(x)
        val_ds.transform = lambda x: trainer.model.get_image_features(x)
        
        print("Training on classes", train_ds.get_active_class_names())

        trainer.init_opt(config)
        for e in range(config.epochs):
            if with_wandb:
                wandb.log({'scale_scheduler_lr':trainer.scale_scheduler.get_lr()[0]})
                wandb.log({'meta_scheduler_lr':trainer.meta_scheduler.get_lr()[0]})
            
            train_stats = []
            for batch in tqdm(train_sampler, desc=f"Training epoch {e} on step {i}"):
                t_stats = trainer.train_step(
                    batch,
                    lwf_beta
                )
                train_stats.append(t_stats)

            train_stats = avg_performance_metrics(train_stats)

            print("Training stats", train_stats)

            if with_wandb:
                train_stats = {f"train_step_{i}/{k}":v for k,v in train_stats.items()}
                wandb.log(train_stats)

            trainer.schedule_step()
            val_stats = trainer.test(val_sampler)
            val_stats_full = trainer.test(val_sampler_full)

            if with_wandb:
                val_stats_full = {f"val_full_step_{i}/{k}":v for k,v in val_stats_full.items()}
                val_stats = {f"val_step_{i}/{k}":v for k,v in val_stats.items()}
                wandb.log(val_stats)
                wandb.log(val_stats_full)

            print("Validation stats", val_stats)
            
        trainer.prev_model = copy.deepcopy(trainer.model)


if __name__ == '__main__':
    with wandb.init(project='CoCoCoOp', config=this_config):
        run(this_config, True)
    


