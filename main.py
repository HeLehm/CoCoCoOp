from torchvision.transforms import ToTensor

from src.models import CoCoCoOp, CachedTextEmbedder
from src.datasets import Flowers102, DatasetSampler
from src.utils import avg_performance_metrics

EPOCHS = 10

ds = Flowers102(split='train')
ds.one_hot_encode_labels()
sampler = DatasetSampler(ds, 1, 16)

ds_val = Flowers102(split='val')
ds_val.one_hot_encode_labels()
val_sampler = DatasetSampler(ds_val, 1, 1)


model = CoCoCoOp()
model.build_model(ds.get_class_names(), clip_model_name='ViT-B/16')
cached_text_embedder = model.create_cached_text_embedder()

ds.transform = model.clip_img_preprocess
ds_val.transform = model.clip_img_preprocess

b = 0
model.start_training()
for e in range(EPOCHS):
    stats = []
    for batch in sampler:
        model.model.train()
        batch_stats = model.forward_backward(batch)
        stats.append(batch_stats)
    stats = avg_performance_metrics(stats)
    print(f'Epoch {e} stats: {stats}')

    val_acc = model.test(val_sampler)
    print(val_acc)
        


