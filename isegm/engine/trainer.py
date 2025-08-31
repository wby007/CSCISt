import os
import logging
from copy import deepcopy
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import loralib as lora

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.vis import draw_probmap, draw_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.serialization import get_config_repr
from isegm.utils.distributed import get_dp_wrapper, get_sampler, reduce_loss_dict
from .optimizer import get_optimizer, get_optimizer_with_layerwise_decay


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 layerwise_decay=False,
                 image_dump_interval=200,
                 checkpoint_interval=10,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 click_models=None,
                 prev_mask_drop_prob=0.0,
                 use_category_labels=True,  # 启用类别标签
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.click_models = click_models
        self.prev_mask_drop_prob = prev_mask_drop_prob
        self.use_category_labels = use_category_labels

        self.enable_lora = cfg.enable_lora
        self.lora_switch_epoch = cfg.lora_switch_epoch
        self.current_epoch = cfg.start_epoch

        # 分布式训练配置
        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        # 指标配置
        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        # 数据集加载
        self.trainset = trainset
        self.valset = valset
        logger.info(f'Training samples: {trainset.get_samples_number()}')
        logger.info(f'Validation samples: {valset.get_samples_number()}')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        # 优化器配置
        if layerwise_decay:
            self.optim = get_optimizer_with_layerwise_decay(model, optimizer, optimizer_params)
        else:
            self.optim = get_optimizer(model, optimizer, optimizer_params)
        model = self._load_weights(model)

        # 多GPU配置
        if cfg.multi_gpu:
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        if self.is_master:
            logger.info(model)
            logger.info(get_config_repr(model._config))

        self.device = cfg.device
        self.net = model.to(self.device)
        self.lr = optimizer_params['lr']

        # LoRA配置
        self.white_list = None
        if self.enable_lora:
            lora.mark_only_lora_as_trainable(self.net)
            self.white_list = ["foreground_embeds", "background_embeds",
                                "text_align_mlp", "collaborative_attn",
                                "neck", "head"   , "phrase2gra_proj"
                               ]

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, list):
                self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optim,
                    schedulers=[s(optimizer=self.optim) for s in lr_scheduler],
                    milestones=[55]
                )
            else:
                self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        # 点击模型配置（如果有）
        if self.click_models is not None:
            for click_model in self.click_models:
                click_model.to(self.device)
                click_model.eval()

    @property
    def is_master(self):
        return not self.cfg.distributed or self.cfg.local_rank == 0

    def _load_weights(self, model):
        if self.cfg.weights is not None and os.path.exists(self.cfg.weights):
            logger.info(f'Loading weights from {self.cfg.weights}')
            checkpoint = torch.load(self.cfg.weights, map_location='cpu')
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        return model

    def _update_lora_training_mode(self):

        if not self.enable_lora or self.lora_switch_epoch is None:
            return



    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch
        self.current_epoch = start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            self._update_lora_training_mode()
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100) \
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, epoch=epoch)

            loss.backward()
            self.optim.step()

            losses_logging['overall'] = loss
            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.is_master:

                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)


                if self.image_dump_interval > 0 and global_step % self.image_dump_interval == 0:
                    self.save_visualization(splitted_batch_data, outputs, global_step, prefix='train')

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else
                                   self.lr_scheduler.get_last_lr()[-1],
                                   global_step=global_step)

                tbar.set_description(f'Epoch {epoch}, training loss {train_loss / (i + 1):.4f}')

        if self.is_master:

            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                            epoch=None, multi_gpu=self.cfg.multi_gpu, save_lora=self.enable_lora,
                            white_list=self.white_list)

            if epoch % self.checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu, save_lora=self.enable_lora,
                                white_list=self.white_list)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()

        if self.is_master:
            avg_val_loss = val_loss / len(self.val_data)
            self.sw.add_scalar(tag=f'{log_prefix}Losses/overall',
                               value=avg_val_loss,
                               global_step=epoch)

            logger.info(f'Epoch {epoch}, validation loss {avg_val_loss:.4f}')

    def batch_forward(self, batch_data, training=True, epoch=None):

        images = batch_data['images'].to(self.device, dtype=torch.float32)
        points = batch_data['points'].to(self.device, dtype=torch.float32)
        masks = batch_data.get('masks', None)
        if masks is not None:
            masks = masks.to(self.device, dtype=torch.float32)

        category_label = batch_data.get('category_labels', None)


        outputs = self.net(
            image=images,
            points=points,
            category_label=category_label
        )

        loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
        losses_logging = {'overall': loss}

        if 'loss_nfl' in outputs:
            losses_logging['nfl'] = outputs['loss_nfl']
        if 'loss_con' in outputs:
            losses_logging['con'] = outputs['loss_con']
        if 'loss_dis' in outputs:
            losses_logging['dis'] = outputs['loss_dis']

        splitted_batch_data = {
            'images': images,
            'points': points,
            'masks': masks,
            'category_label': category_label
        }

        return loss, losses_logging, splitted_batch_data, outputs

    def save_visualization(self, batch_data, outputs, step, prefix='train'):

        img = batch_data['images'][0].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        mask = batch_data['masks'][0].cpu().numpy().squeeze() if batch_data['masks'] is not None else None
        pred = torch.sigmoid(outputs['instances'][0]).cpu().numpy().squeeze()

        vis_list = [img]
        vis_list.append(draw_probmap(pred, img.shape[:2]))
        vis_list.append(draw_points(img.shape[:2], batch_data['points'][0].cpu().numpy()))

        if mask is not None:
            vis_list.append(draw_probmap(mask, img.shape[:2]))

        vis = np.concatenate(vis_list, axis=1)
        self.sw.add_image(f'{prefix}/visualization', vis.transpose(2, 0, 1), step)

