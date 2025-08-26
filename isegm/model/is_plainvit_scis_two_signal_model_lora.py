from isegm.utils.serialization import serialize
from .is_model import ISModel
from .is_plainvit_scis_model_lora import SimpleFPN
from .modeling.models_vit_dual_lora import VisionTransformer_duallora, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead
from .modeling.clip_text_encoding import ClipTextEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollaborativeAttention(nn.Module):


    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5  # 缩放因子 sqrt(d_k)

        self.q_img = nn.Linear(d_model, d_model)
        self.k_prompt = nn.Linear(d_model, d_model)
        self.v_prompt = nn.Linear(d_model, d_model)

        self.q_prompt = nn.Linear(d_model, d_model)
        self.k_img = nn.Linear(d_model, d_model)
        self.v_img = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, img_feat, prompt_feat):

        B, N_img, C = img_feat.shape
        N_prompt = prompt_feat.shape[1]

        q_img = self.q_img(img_feat).view(B, N_img, self.nhead, C // self.nhead).transpose(1, 2)
        k_prompt = self.k_prompt(prompt_feat).view(B, N_prompt, self.nhead, C // self.nhead).transpose(1, 2)
        v_prompt = self.v_prompt(prompt_feat).view(B, N_prompt, self.nhead, C // self.nhead).transpose(1, 2)

        attn_img2prompt = (q_img @ k_prompt.transpose(-2, -1)) * self.scale
        attn_img2prompt = F.softmax(attn_img2prompt, dim=-1)
        ca_img2prompt = (attn_img2prompt @ v_prompt).transpose(1, 2).contiguous().view(B, N_img, C)

        q_prompt = self.q_prompt(prompt_feat).view(B, N_prompt, self.nhead, C // self.nhead).transpose(1, 2)
        k_img = self.k_img(img_feat).view(B, N_img, self.nhead, C // self.nhead).transpose(1, 2)
        v_img = self.v_img(img_feat).view(B, N_img, self.nhead, C // self.nhead).transpose(1, 2)

        attn_prompt2img = (q_prompt @ k_img.transpose(-2, -1)) * self.scale
        attn_prompt2img = F.softmax(attn_prompt2img, dim=-1)
        ca_prompt2img = (attn_prompt2img @ v_img).transpose(1, 2).contiguous().view(B, N_prompt, C)

        fused_img = torch.cat([ca_img2prompt, img_feat], dim=-1)
        fused_img = self.proj(fused_img)

        return fused_img


class CrossModalPriorFusion(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, click_feat, text_feat):
        # 对齐特征长度
        text_feat_expanded = text_feat.repeat(1, click_feat.shape[1], 1)
        # 融合
        fused = torch.cat([click_feat, text_feat_expanded], dim=-1)
        fused = self.mlp(fused)
        return self.norm(fused)


@serialize
class PhraseCLIPscisModel_lora(ISModel):
    def __init__(
            self,
            backbone_params={},
            phrase_encoder_params={},
            neck_params={},
            head_params={},
            random_split=False,
            num_foreground_prompts=4,
            num_background_prompts=2,
            text_mlp_dim=512,
            contrastive_tau=0.07,
            alpha=1.0,
            beta=0.2,
            gamma=0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.random_split = random_split
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.contrastive_tau = contrastive_tau

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer_duallora(**backbone_params)
        self.phrase_encoder = ClipTextEncoder(**phrase_encoder_params)  # CLIP文本编码器 H_T
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

        self.text_align_mlp = nn.Sequential(
            nn.Linear(self.phrase_encoder.out_dim, text_mlp_dim),
            nn.ReLU(),
            nn.Linear(text_mlp_dim, backbone_params['embed_dim'])
        )

        self.foreground_embeds = nn.Parameter(
            torch.empty(num_foreground_prompts, backbone_params['embed_dim'])
        )

        self.background_embeds = nn.Parameter(
            torch.empty(num_background_prompts, backbone_params['embed_dim'])
        )
        nn.init.xavier_uniform_(self.foreground_embeds)
        nn.init.xavier_uniform_(self.background_embeds)

        # 跨模态
        self.cross_modal_fusion = CrossModalPriorFusion(
            2 * backbone_params['embed_dim'],
            backbone_params['embed_dim']
        )

        self.collaborative_attn = CollaborativeAttention(
            d_model=backbone_params['embed_dim'],
            nhead=backbone_params.get('nhead', 8)
        )

    def forward(self, image, points, mask=None, gra=None, text=None, category_label=None):
        # 确保所有参数都传递给backbone_forward
        outputs = self.backbone_forward(image, points, gra=gra, text=text, category_label=category_label)
        instances = outputs['instances']

        if mask is not None and category_label is not None:
            loss_nfl = self.normalized_focal_loss(instances, mask)
            interaction_embeds = self._get_interaction_embeddings(points, mask)
            prompt_embeds = self._get_prompt_embeddings(category_label)
            loss_con = self.contrastive_loss(interaction_embeds, prompt_embeds)
            ft, bt = self._get_foreground_background_features(mask)
            loss_dis = self.discriminative_loss(ft, bt)
            total_loss = self.alpha * loss_nfl + self.beta * loss_con + self.gamma * loss_dis
            return {'instances': instances, 'loss': total_loss}

        return outputs

    def backbone_forward(self, image, coord_features=None, gra=None, text=None, category_label=None):
        # 确保参数互斥
        input_count = sum(1 for x in [gra, text, category_label] if x is not None)
        if input_count > 1:
            raise ValueError("gra, text, category_label cannot be used simultaneously")

        # 1. 处理用户点击+历史掩码分支
        click_features = self.patch_embed_coords(coord_features)  # P_click+prev
        B, N_click, C = click_features.shape

        # 2. 处理标签文本分支
        text_prompt_features = None
        if category_label is not None:
            # 生成文本提示 P_text
            text_prompt_features = self._get_prompt_embeddings(category_label)  # [B, N_p+N_b, C]
            # 跨模态先验融合
            click_features = self.cross_modal_fusion(click_features, text_prompt_features)

        # 3. 获取backbone特征
        backbone_features = self.backbone.forward_backbone(
            image,
            click_features,
            gra=gra,  # 传递粒度参数
            text_features=text,  # 传递文本参数
            shuffle=self.random_split
        )
        print(f"backbone_features shape after backbone: {backbone_features.shape}")  # 新增

        # 4. 应用协作注意力机制
        if text_prompt_features is not None:
            backbone_features = self.collaborative_attn(backbone_features, text_prompt_features)
            print(f"backbone_features shape after collaborative_attn: {backbone_features.shape}")  # 新增

        # 5. 提取多尺度特征并输出
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size
        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        print(f"backbone_features shape before neck: {backbone_features.shape}")  # 新增
        multi_scale_features = self.neck(backbone_features)
        print(f"multi_scale_features from neck: {[f.shape for f in multi_scale_features]}")  # 新增（如果是列表）

        return {'instances': self.head(multi_scale_features), 'instances_aux': None}

    def _get_prompt_embeddings(self, category_label):
        """生成文本提示嵌入 P_text"""
        B = len(category_label)
        # 统一模板 "a photo of a [CLS]"
        cls_texts = [f"a photo of a {label}" for label in category_label]
        # CLIP文本编码 H_T(M^d)
        cls_features = self.phrase_encoder(cls_texts)  # [B, clip_dim]
        # 文本特征对齐 H_M处理
        aligned_text_feat = self.text_align_mlp(cls_features)  # [B, C]

        # 生成前景提示 p_i = H_M(H_T(M^d)) + t_i
        foreground_prompts = aligned_text_feat.unsqueeze(1) + self.foreground_embeds  # [B, N_p, C]
        # 生成背景提示 B
        background_prompts = self.background_embeds.unsqueeze(0).repeat(B, 1, 1)  # [B, N_b, C]
        # 拼接为 P_text
        return torch.cat([foreground_prompts, background_prompts], dim=1)  # [B, N_p+N_b, C]

    def _get_interaction_embeddings(self, points, mask):

        click_embeds = self.patch_embed_coords(points)

        prev_mask_embeds = F.adaptive_avg_pool2d(mask, (click_embeds.shape[1], 1)).squeeze(-1)
        return torch.cat([click_embeds, prev_mask_embeds], dim=1)

    def contrastive_loss(self, interaction_embeds, prompt_embeds):

        B, T, C = interaction_embeds.shape
        # 提示嵌入与交互嵌入对齐
        prompt_embeds = F.adaptive_avg_pool1d(
            prompt_embeds.transpose(1, 2), T
        ).transpose(1, 2)

        # 计算相似度
        sim = torch.cosine_similarity(interaction_embeds, prompt_embeds, dim=-1) / self.contrastive_tau
        # 对比损失计算
        return -torch.mean(torch.log(F.softmax(sim, dim=-1).diag()))

    def normalized_focal_loss(self, pred, target, gamma=2.0):

        pred = torch.sigmoid(pred)
        target = target.float()

        pt = (1 - pred) * target + pred * (1 - target)
        focal_weight = (1 - pt) ** gamma
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        focal_loss = focal_weight * bce

        norm_factor = focal_weight.sum() + 1e-8
        return (focal_loss.sum() / norm_factor).mean()

    def _get_foreground_background_features(self, mask):

        B, L, C = self.backbone.last_features.shape
        H, W = self.backbone.patch_embed.grid_size
        # 调整掩码尺寸
        mask_reshaped = F.interpolate(mask, size=(H, W), mode='bilinear').view(B, 1, H * W)  # [B, 1, H*W]

        ft = self.backbone.last_features * mask_reshaped.transpose(1, 2)
        bt = self.backbone.last_features * (1 - mask_reshaped.transpose(1, 2))
        return ft, bt

    def discriminative_loss(self, ft, bt):
        B, T, C = ft.shape

        ft_mean = ft.mean(dim=1, keepdim=True)
        loss_ft = ((ft - ft_mean) ** 2).mean()
        bt_mean = bt.mean(dim=1, keepdim=True)
        loss_bt = ((bt - bt_mean) ** 2).mean()

        loss_sep = ((ft_mean - bt_mean) ** 2).mean()

        return loss_ft + loss_bt + loss_sep

