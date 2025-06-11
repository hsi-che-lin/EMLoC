# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import os
import warnings
import torch.distributed as dist
from transformers import Trainer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from internvl.data.build import build_datasets
from internvl.model.build import build_tokenizer_and_model
from internvl.utils.dist_utils import init_dist
from internvl.patch import monkey_patch
from internvl.utils.args import getArgs
from internvl.utils.utils import initLogger
from internvl.engine.finetune_utils.loraCorrection import loraCorrection


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher = launcher, backend = 'nccl')
    model_args, data_args, training_args = getArgs()
    monkey_patch(data_args.use_packed_ds)
    initLogger(training_args, logger)
    
    # Detecting last checkpoint and eventually continue from last checkpoint.
    if (training_args.resume_from_checkpoint is not None):
        checkpoint = training_args.resume_from_checkpoint
    elif (os.path.isdir(training_args.output_dir) and (not training_args.overwrite_output_dir)):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        
        if ((checkpoint is None) and (len(os.listdir(training_args.output_dir)) > 0)):
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif (checkpoint is not None):
            logger.info(
                f'Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    else:
        checkpoint = None
    
    # model
    set_seed(training_args.seed)
    logDir = os.path.join(training_args.output_dir, "log-model") if (dist.get_rank() == 0) else None
    tokenizer, model = build_tokenizer_and_model(model_args, data_args, logDir)

    # data
    set_seed(training_args.seed)
    train_dataset, collator = build_datasets(
        data_args,
        tokenizer,
        model,
        train_batch_size = training_args.train_batch_size,
        group_by_length = training_args.group_by_length
    )

    # Training
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = None,
        tokenizer = tokenizer,
        data_collator = collator,
    )
    trainer.train(resume_from_checkpoint = checkpoint)
    loraPath = f"{get_last_checkpoint(training_args.output_dir)}/adapter_model.safetensors"
    loraCorrection(model_args.model_name_or_path, loraPath)


if (__name__ == "__main__"):
    main()
