"""
关于训练框架的说明：

本项目基于 PyTorch Lightning 框架开发，整体设计力求贴合 Lightning 的设计哲学。

需特别说明的是，由于本框架搭建于早期学习阶段，受限于当时的技术视野，部分实现细节尚显稚嫩，未达到最佳实践标准（例如：控制台输出主要依赖 print 而非更规范的 logger 模块）。

虽有不足，但希望能为各位提供参考。代码中的不成熟之处，恳请各位同仁海涵与指正。

===English:

Note on the Training Architecture:

This project utilizes PyTorch Lightning and strives to adhere to its core design philosophy.

Please note that this framework was established during an earlier stage of my development journey. Consequently, certain implementation details may not align with current best practices (for instance, console outputs rely heavily on print rather than the standard logger).

I ask for your understanding regarding these imperfections. Any feedback or suggestions for improvement from the community would be greatly appreciated.
"""

import os
import argparse

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.strategies import DDPStrategy

import utils
from method import setup_model
from data import setup_data

def default_parser():
    parser = argparse.ArgumentParser(description='FocusST Default Parameters.')

    # Tasks Options(任务配置): Priority(优先级): --ckpt > --args > --conf
    parser.add_argument('--ckpt', type=str, default='', help='./test.ckpt : Continue training ckpt file.')
    parser.add_argument('--args', type=str, default='', help='./args.yaml : Using an existing args file.')
    parser.add_argument('--conf', type=str, default='', help='./data.conf : Dataset configuration.')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode.')

    # Multitask
    # 对于想跑一堆实验的同学，相比这个功能，我更推荐使用本项目提供的 batch_runner.py 脚本（表现更加稳定，亲测好用！）。
    # 该脚本支持实验自动排队，遇到报错自动跳过并继续执行后续任务。
    # For researchers needing to run a large number of experiments, I highly recommend using the included batch_runner.py script.
    # It features automatic queuing and a fault-tolerance mechanism (automatically skipping failed runs to proceed with the next). I have personally tested this tool and found it to be extremely efficient and reliable for managing extensive workloads.
    parser.add_argument('--batch_args_path', type=str, default='',
                        help='Directory containing args.yaml files for batch jobs.', )

    # Working Options(项目配置):
    parser.add_argument('--working_path', type=str, default='working_path', help='Program working path.')
    parser.add_argument('--project_name', type=str, default='mnist', help='/.../working_path/project_name.')
    parser.add_argument('--version_name', type=str, default='debug', help='/.../working_path/version_name.')

    # Method Options(模型配置):
    parser.add_argument('--model', type=str, default='LaST', help='Method Name.',
                        choices=['DEMO', 'LaST', 'PredFormer'])
    parser.add_argument('--d_model', type=str, default=128, help='Hidden layer dimensions of the model.')
    parser.add_argument('--s_patch', type=int, default=2, help='Size of patch(Pixel shuffle).')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_coder', type=int, default=8, help='Number of encoder & decoder layers')
    parser.add_argument('--n_coder_down', type=int, default=0, help='Number of encoder & decoder layers')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of translator layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Droppath rate.')
    parser.add_argument('--s_local', type=bool, nargs='*', default=[3, 3, 3], help='Local attention window size')
    parser.add_argument('--r_forward', type=int, default=4, help='FFN ratio')
    parser.add_argument('--s_kernel', type=int, default=1, help='Encoder/Decoder kernel')
    parser.add_argument('--attn_bias', action='store_true', default=False, help='Use bias in attention')
    
    # Lightning Framework(Lightning框架配置):
    parser.add_argument('--seed', type=int, default=42, help='Random seed (lightning.seed_everything).')
    parser.add_argument('--device', default='gpu', type=str, choices=['gpu', 'cpu'], help='config accelerator')
    parser.add_argument('--n_device', default=1, type=int, help='Number of gpus on the current node.')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 (mixed precision)')
    parser.add_argument('--n_val_every', default=5, type=int, help='validation every n epochs')
    parser.add_argument('--logger_wandb', action='store_true', default=False, help='Enable wandb logger.')
    parser.add_argument('--n_model_save', type=int, default=5, help='Enable wandb logger.')
    parser.add_argument('--flops', action='store_true', default=False, help='check model flops')

    # Training Configuration(训练配置)
    parser.add_argument('--epoch', default=200, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--gard_clip', default=None, type=float, help='')
    parser.add_argument('--lr_scheduler', default='cos', choices=['cos', 'onecycle'], type=str)

    # Lightning - Distributed training(分布式训练)
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Use DDP, base on linux system, Here is an example of two hosts with one GPU per host:'
                             'MASTER_ADDR=192.168.31.100 MASTER_PORT=10109 WORLD_SIZE=2 NODE_RANK=2 python main.py ')
    parser.add_argument('--num_nodes', default=1, type=int, help='The number of nodes used to DDP training')

    # Dataset Configuration(数据集配置): Most of the datasets support automatic download(大部分数据集支持自动下载)
    parser.add_argument('--data_root', type=str, default='data', help='Dataset storage location.')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='Training set batch size.')
    parser.add_argument('--val_batch_size', '-vb', default=32, type=int, help='Validation set batch size.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Specifies the number of workers for loading data'
                             'Due to limitations with multiprocessing on Windows, set 0 when running on Windows.')
    parser.add_argument('--accumulate_grad_batches', default=1, type=int, help='Gradient accumulation.')
    parser.add_argument('--model_ckpt', type=str, default='', help='Model checkpoint path.')

    # If --conf is configured, these arguments will be replaced(如果配置了--conf，那么配置文件将会替换这些参数)
    parser.add_argument('--data', type=str, default='TaxiBJ', help='Dataset name.',
                        choices=['TaxiBJ', 'Human', 'KTH', 'MovingMNIST', 'SeaWaveCora', 'sevirlr', 't2m', 'r', 'tcc', 'uv10', 'KittiCaltech'])
    parser.add_argument('--metrics', nargs='+', default=['mae', 'mse'], help='Metrics to evaluate.',
                        choices=['mae', 'mse', 'psnr', 'ssim', 'mae_pixel', 'mse_pixel'])
    parser.add_argument('--data_back', type=int, nargs=4, default=[10, 1, 64, 64], help="[len_back,C,H,W]")
    parser.add_argument('--data_pred', type=int, nargs=4, default=[10, 1, 64, 64], help="[len_pred,C,H,W]")
    parser.add_argument('--data_val_rate', type=float, default=0.2, help='Proportion of validation set.')
    parser.add_argument('--data_test_rate', type=float, default=0.1, help='Proportion of test set.')
    parser.add_argument('--data_mean', type=float, nargs='*', default=[0.0], help='Mean of dataset.)')
    parser.add_argument('--data_std', type=float, nargs='*', default=[255.0], help='Standard deviation of data.')
    parser.add_argument('--draw_style', type=str, default='gray', choices=['gray', 'heat'], help='Drawing style.')
    parser.add_argument('--draw_per_channel', type=bool, default=False, help='Drawing per channel.')
    parser.add_argument('--inverse_transform', type=bool, default=False, help='Inverse transformation.')
    parser.add_argument('--heat_cmap', type=str, default="viridis")  # 'GnBu' viridis

    return parser


def run(parser):
    # Configuring Experimental Parameters(配置实验参数)
    args = utils.setup_experiment(parser)

    # Fixed random seed(固定随机种子)
    L.seed_everything(args.seed)

    # Prepare Lightning Data Module(准备数据模块)
    data = setup_data(args)

    # Prepare Lightning Model Module(准备模型)
    model = setup_model(args)

    # Prepare Lightning Logger
    loggers = [utils.logger.CustomCSVLogger(
        save_dir=args.working_path,
        name=args.project_name,
        version=args.version,
        version_dir=args.version_path
    )]
    if args.logger_wandb:  # Logger: Weights and biases - www.wandb.ai
        wandb_logger = utils.logger.CustomWandbLogger(
            save_dir=args.version_path,
            project=args.project_name,
            name=args.version_name,
            version_id=args.version,
        )
        wandb_logger.watch(model, log="all")
        loggers.append(wandb_logger)

    # Prepare Lightning Callbacks:
    callbacks = [ModelCheckpoint(
        dirpath=str(os.path.join(args.version_path, 'checkpoints')),
        save_top_k=args.n_model_save,
        save_last=True,
        monitor='val_loss',
        mode='min',
        filename="{epoch:04d}-{val_loss:.6f}"
    ),
        utils.callbacks.LogValidationMetric(metrics=args.metrics, inverse_transform=args.inverse_transform,
                                            data_mean=args.data_mean, data_std=args.data_std),
        utils.callbacks.LogLearningRate(),
        utils.callbacks.SaveConfigs(args=args, save_path=args.version_path),
        utils.callbacks.PrintConfig(args),
        utils.callbacks.CustomRichProgressBar(),
        RichModelSummary(max_depth=2),
        utils.callbacks.Visualizer(
            style=args.draw_style, i_image=0, data_mean=args.data_mean, data_std=args.data_std, n_val_img=10,
            n_test_img=3,
                save_dir=args.version_path,
            save_png=True, save_gif=False,
            draw_per_channel=args.draw_per_channel,
            heat_cmap=args.heat_cmap,
        )
    ]
    if args.flops:
        callbacks.append(utils.callbacks.CheckFLIOS(args.data_back, model))

    # Prepare Lightning Trainer
    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        # fast_dev_run=3,
        num_sanity_val_steps=1,  # 训练前验证1次，保证模型无错误
        accelerator=args.device,
        devices=args.n_device,
        strategy=DDPStrategy(find_unused_parameters=False) if args.ddp else 'auto',
        num_nodes=args.num_nodes,
        precision='16-mixed' if args.fp16 else '32-true',  # 16-mixed
        default_root_dir=args.version_path,
        max_epochs=args.epoch,
        check_val_every_n_epoch=args.n_val_every,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=loggers,
        callbacks=callbacks,
        enable_model_summary=False,
        # profiler="simple",
        gradient_clip_val=args.gard_clip,

    )

    if args.eval:
        trainer.test(model=model, ckpt_path=args.ckpt, datamodule=data)
    else:
        # Start training:
        trainer.fit(
            model=model,
            datamodule=data,
            ckpt_path=args.ckpt if args.ckpt != '' else None,
        )


if __name__ == '__main__':
    default_parser = default_parser()
    initial_args = default_parser.parse_args()

    if initial_args.batch_args_path != '':
        utils.setup_multitask(initial_args, default_parser, run)  # Multitask(Batch job functionality) 多任务
    else:
        run(default_parser)  # Normal tasks(正常训练任务)

