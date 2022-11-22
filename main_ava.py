from torch.optim import SGD
import torch
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from avalanche.models import SimpleMLP

from src.models import CoCoCoOp
from src.utils import load_clip, load_class_order, load_class_names

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
clip_backbone = 'ViT-B/16'
clip_model, clip_preprocess = load_clip(clip_backbone)


# src/Submodules/ConCLIP/dataset_reqs/tinyimagenet_classes.txt

#fixed_class_order = load_class_order('./src/Submodules/ConCLIP/class_orders/tinyimagenet.yaml')
class_names = load_class_names('./src/Submodules/ConCLIP/dataset_reqs/tinyimagenet_classes.txt')

scenario = SplitTinyImageNet(
    n_experiences=5,
    seed=42,
    train_transform=clip_preprocess,
    eval_transform=clip_preprocess,
    fixed_class_order=[i for i in range(200)],
    dataset_root='./src/datasets/data/',
)

# buidl model

model = CoCoCoOp(
    clip_model,
    ctx_init = 'a photo of a',
    device = device,
)


# logging
wandb_logger = WandBLogger(project='CoCoCoOp', entity="bschergen")
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[wandb_logger, interactive_logger]
)

meta_strategy = Naive( # TODO: LwF
    model, SGD(model.prompt_learner.meta_net.parameters(), lr=0.001, momentum=0.9), CrossEntropyLoss(),
    train_mb_size=4, train_epochs=1, eval_mb_size=16, evaluator=eval_plugin
)

train_results = []
test_results = []
for experience in scenario.train_stream:

    class_names_in_exp = [class_names[i] for i in experience.classes_in_this_experience]
    model.add_class_names(class_names_in_exp)

    t_res = meta_strategy.train(experience)
    train_results.append(t_res)

    print(t_res)

    test_result = meta_strategy.eval(scenario.test_stream)
    test_results.append(test_result)

    print(test_result)



    
