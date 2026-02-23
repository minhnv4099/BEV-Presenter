# Copyright (c) OpenMMLab. All rights reserved.
"""MMENGINEection3D provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in mmengine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
# from mmengine.registry import INFERENCERS as MMENGINE_INFERENCERS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry, build_from_cfg
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTIONS

PACKAGE = 'src.bevformer'

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner',
    parent=MMENGINE_RUNNERS, locations=[f'src.runner'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS, locations=[f'src.runner'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop',
    parent=MMENGINE_LOOPS, locations=[f'src.runner'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['src.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=[f'src.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler', parent=MMENGINE_DATA_SAMPLERS,
    locations=[f'src.datasets.samplers'])
TRANSFORMS = Registry(
    'transform', parent=MMENGINE_TRANSFORMS,
    locations=["src.datasets.transforms"])
PIPELINES = Registry(
    "pipeline", scope="pipeline", parent=None,
    locations=["src.datasets.transforms"])

CONFIGS = Registry(
    'config', locations=[f'{PACKAGE}.models'])
# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=[f'{PACKAGE}.models'])
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=[f'{PACKAGE}.models'])
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=[f'{PACKAGE}.models'])

# manage all kinds of transformer components
ATTENTIONS = Registry(
    'attention', scope='attention', parent=MODELS,
    locations=[f'{PACKAGE}.models'])
TRANSFORMER_LAYERS = Registry(
    "transformer_layer", scope="transformer_layer", parent=MODELS,
    locations=[f'{PACKAGE}.models'])
TRANSFORMER_BLOCKS = Registry(
    "transformer_block", scope="transformer_block", parent=MODELS,
    locations=[f'{PACKAGE}.models'])
TRANSFORMERS = Registry(
    "transformer", scope="transformer", parent=MODELS,
    locations=[f'{PACKAGE}.models'])
POSITION_ENCODINGS = Registry(
    'position_encoding', scope='position_encoding', parent=MODELS,
    locations=[f'{PACKAGE}.models', f'{PACKAGE}.models.layers'])

ATTENTIONS = MODELS
TRANSFORMER_LAYERS = MODELS
TRANSFORMER_BLOCKS = MODELS
TRANSFORMERS = MODELS
POSITION_ENCODINGS = MODELS

# manage all kinds of extractors
BACKBONES = Registry(
    "backbone", scope="backbone", parent=MODELS,
    locations=[f'{PACKAGE}.models', f'{PACKAGE}.models.backbones'])
NECKS = Registry(
    "neck", scope="neck", parent=MODELS,
    locations=[f'{PACKAGE}.models', f'{PACKAGE}.models.necks'])
# manage all kinds of heads
HEADS = Registry(
    "head", scope="head", parent=MODELS, locations=[f'{PACKAGE}.models'])
# manage all kinds of detectors
DETECTORS = Registry(
    "detector", scope="detector", parent=MODELS,
    locations=[f'{PACKAGE}.models', f'{PACKAGE}.models.detectors'])

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
DETECTORS = MODELS

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=[f'{PACKAGE}.models'])
BBOX_CODERS = Registry(
    "bbox_coder", scope="bbox_coder",
    locations=[f'{PACKAGE}.core.bbox'])
BBOX_SAMPLERS = Registry(
    "bbox_sampler", scope="bbox_sampler",
    locations=[f'src.bevformer.core.bbox.samplers'])
ASSIGNERS = Registry(
    "assigner", scope="assigner",
    locations=[f'{PACKAGE}.core.bbox'])
LOSSES = Registry(
    "loss", scope="loss",
    locations=[f'{PACKAGE}.models', f'{PACKAGE}.models.losses'])
MATCH_COST = Registry(
    "match_cost", scope="match_cost",
    locations=[f'{PACKAGE}.core.bbox'])

# BBOX_CODERS = TASK_UTILS
# BBOX_SAMPLERS = TASK_UTILS
# ASSIGNERS = TASK_UTILS
# LOSSES = TASK_UTILS
# MATCH_COST = TASK_UTILS

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer', parent=MMENGINE_OPTIMIZERS,
    locations=[f'src.bevformer.models.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=[f'src.bevformer.models.optimizers'])
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=[f'{PACKAGE}.engine'])
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['src.evaluation.metrics'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['src.evaluation'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS)

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS)

FUNCTIONS = Registry(
    'function', parent=MMENGINE_FUNCTIONS,
    locations=['src.datasets'])
# manage inferencer
# INFERENCERS = Registry(
#     'inferencer',
#     parent=MMENGINE_INFERENCERS,
#     locations=[f'{PACKAGE}.api.inferencers'])
