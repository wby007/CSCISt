# 从默认配置导入相关内容（可能包含一些通用的配置、函数等）
from isegm.utils.exp_imports.default import *
# 导入交叉熵损失函数
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

# 模型名称
MODEL_NAME = 'sbd_plainvit_base448'


# 主函数，用于初始化模型并进行训练
def main(cfg):
    # 初始化模型和获取模型配置
    model, model_cfg = init_model(cfg)
    # 调用训练函数
    train(model, cfg, model_cfg)

# 初始化模型函数
def init_model(cfg):
    model_cfg = edict()
    # 设置裁剪尺寸
    model_cfg.crop_size = (448, 448)
    # 设置最大点数
    model_cfg.num_max_points = 24

    # 定义骨干网络参数
    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    # 定义颈部参数
    neck_params = dict(
        in_dim=768,
        out_dims=[128, 256, 512, 1024],
    )

    # 定义头部参数
    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1': 256, 'x2': 128, 'x4': 64}[cfg.upsample]
    )

    # 创建Plain Vit模型实例
    model = PlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )

    # 从预训练模型初始化骨干网络权重
    model.backbone.init_weights_from_pretrained(cfg.IMAGENET_PRETRAINED_MODELS.MAE_BASE)
    # 将模型移动到指定设备（如GPU）
    model.to(cfg.device)

    return model, model_cfg

# 训练函数
def train(model, cfg, model_cfg):
    """
    训练模型的主函数。

    参数:
    - model: 要训练的模型。
    - cfg: 包含训练配置的字典。
    - model_cfg: 包含模型特定配置的字典。
    """
    # 如果批大小小于1，则设置为32，否则使用传入的批大小，同时设置验证批大小
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    # 损失函数配置
    loss_cfg = edict()
    # 设置实例损失函数和权重
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    # 训练数据增强操作组合
    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    # 验证数据增强操作组合
    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    # 多点采样器
    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    # 训练数据集
    trainset = SBDDataset(
        cfg.SBD_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        samples_scores_path='./assets/sbd_samples_weights.pkl',
        samples_scores_gamma=1.25
    )

    # 验证数据集
    valset = SBDDataset(
        cfg.SBD_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500
    )

    # 优化器参数
    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    # 学习率调度器
    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 55], gamma=0.1)
    # 创建训练器实例
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 20), (50, 1)],
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    # 运行训练（这里先运行1个epoch，注释掉了原本运行55个epoch的代码）
    trainer.run(num_epochs=90, validation=False)
