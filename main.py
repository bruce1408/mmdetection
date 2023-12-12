import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import os.path as osp

import torch
from torch import nn

from mmdet.apis import DetInferencer
from mmdet.utils import setup_cache_size_limit_of_dynamo

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

## 蒸馏，量化
from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.quantizers import AcademicQuantizer


def load_data():
    pass

def load_teacher():
    pass

def load_student():
    pass

def do_distiller(args):
    pass

def do_train(args):
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args['config'])
    cfg.launcher = args['launcher']
    if args['cfg_options'] is not None:
        cfg.merge_from_dict(args['cfg_options'])

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args['work_dir'] is not None:
        # update configs according to CLI args if args['work_dir'] is not None
        cfg.work_dir = args['work_dir']
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args['config']))[0])

    # enable automatic-mixed-precision training
    if args['amp'] is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args['auto_scale_lr']:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args['resume'] == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args['resume'] is not None:
        cfg.resume = True
        cfg.load_from = args['resume']

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

def do_inference(model_name, weights_path, img_path):
    inferencer = DetInferencer(
        model=model_name,
        weights=weights_path,
        device="cpu"
    )

def main():

    model_config = "./yolox_l_distiller_config.py"

    args = {
        'model_config': model_config,
        'save_weights_path' : None,

        'do_train' : True,
        'do_inference' : True,

        'train_args' :{
        "config" : model_config,
        "work_dir": "",
        "amp" : True,
        "auto-scale-lr": True,
        "resume": 'auto',
        "cfg-option":{},
        "launcher": 'pytorch'
        },
    }

    print(args)

    # TODO YOLOx 量化
    # TODO 推理
    # TODO YOLOx 剪枝
    # TODO 推理
    # TODO YOLOx 蒸馏
    # TODO 推理

    if args['do_train']:
        args['save_weights_path'] = do_train(args['train_args'])
    if args['do_inference']:
        do_inference(args['model_name'], args['save_weights_path'])


if __name__ == '__main__':
    main()