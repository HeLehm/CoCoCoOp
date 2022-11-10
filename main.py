from torchvision.transforms import ToTensor

from src.models import CoCoCoOp, CachedTextEmbedder
from src.datasets import Flowers102, DatasetSampler, StanfordCars, Caltech101
from src.utils import avg_performance_metrics
import wandb
from tqdm import tqdm

ds_to_class = {
    'Flowers102': Flowers102,
    'StanfordCars' : StanfordCars,
    'Caltech101': Caltech101,
}

def run_with_config(config):
    with wandb.init(project='CoCoCoOp', config=config):
        config = wandb.config

        ds = ds_to_class[config.Training_ds](split='train', cache_transformed_images=True, download=True)
        ds.one_hot_encode_labels()
        t_sampler = DatasetSampler(ds, config.batch_size, config.per_class)

        v_ds = ds_to_class[config.Training_ds](split='test', cache_transformed_images=True)
        v_ds.one_hot_encode_labels()
        v_sampler = DatasetSampler(v_ds, config.batch_size, config.per_class)


        model = CoCoCoOp() # TODO add cnfig
        model.build_model(ds.get_class_names(), clip_model_name=config.clip_backbone, ctx_init=config.ctx_init)
        model.start_training()

        ds.transform = model.img_to_features
        v_ds.transform = model.img_to_features

        wandb.watch(model.model.prompt_learner)

        for epoch in range(config.epochs):
            model.model.train()
            stats = []
            for batch in tqdm(t_sampler, desc=f"Training epoch {epoch}"):
                batch_stats = model.forward_backward(batch)
                stats.append(batch_stats)
            
            stats = avg_performance_metrics(stats)

            stats = {f'train/{k}': v for k, v in stats.items()}
            wandb.log(stats)

            print(f'Epoch {epoch} train stats: {stats}')

            val_stats = model.test(v_sampler)
            val_stats = {f'val/{k}': v for k, v in val_stats.items()}
            wandb.log(val_stats)


this_config = {
    'Training_ds': 'Caltech101',
    'per_class': 16,
    'batch_size': 1,
    'epochs': 10,
    'lr': 1e-3, # TODOs
    'optimizer': 'Adam', # TODO
    'clip_backbone': 'ViT-B/16',
    'ctx_init': 'a photo of a',
}

if __name__ == '__main__':
    run_with_config(this_config)
        


