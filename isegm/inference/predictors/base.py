import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide


class BasePredictor(object):
    def __init__(self, model, device, sam_type=None,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None, **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.image_features = None  # 特征缓存
        self.device = device
        self.sam_type = sam_type
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        # 图像变换管道
        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        """设置输入图像"""
        if not isinstance(image, torch.Tensor):
            image_nd = self.to_tensor(image)
        else:
            image_nd = image
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_mask=None, category_label=None, gra=None, phrase=None):
        """获取预测结果，支持类别标签和粒度控制输入"""
        clicks_list = clicker.get_clicks()

        # 模型切换（如果有多个点击模型）
        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        # 拼接前一轮掩码（如果模型支持）
        if (hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask) or self.sam_type is not None:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        # 应用图像变换
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )

        # 获取预测结果，传递额外参数
        pred_logits = self._get_prediction(
            image_nd, clicks_lists, is_image_changed,
            category_label=category_label,
            gra=gra,
            phrase=phrase
        )

        # 调整预测尺寸
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        # 逆变换
        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)

        # 处理缩放
        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker, prev_mask, category_label, gra, phrase)

        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists=None, is_image_changed=None,
                        category_label=None, gra=None, phrase=None):
        """获取预测logits"""
        points_nd = self.get_points_nd(clicks_lists)

        # SAM模型处理
        if self.sam_type == 'SAM':
            batched_input = self.get_sam_batched_input(image_nd, points_nd)
            batched_output = self.net(batched_input, multimask_output=False, return_logits=True)
            return torch.cat([batch['masks'] for batch in batched_output], dim=0)

        # 确保输入参数互斥
        input_count = sum(1 for x in [gra, phrase, category_label] if x is not None)
        if input_count > 1:
            raise ValueError("gra, phrase, category_label cannot be used simultaneously")

        # 处理gra参数
        if gra is not None:
            gra = torch.Tensor([gra]).unsqueeze(0).to(self.device)

        # 处理phrase参数
        if phrase is not None:
            if isinstance(phrase, str):
                import clip
                phrase = clip.tokenize(phrase).to(self.device)
            else:
                phrase = phrase.to(self.device)

        # 处理category_label参数
        if category_label is not None:
            if isinstance(category_label, str):
                category_label = [category_label]  # 转为列表形式

        # 根据不同参数组合调用模型
        try:
            # 检查模型是否支持特定参数
            import inspect
            model_forward_params = inspect.signature(self.net.forward).parameters

            # 构建参数字典
            net_input = {
                'image': image_nd,
                'points': points_nd
            }

            # 只添加模型支持的参数
            if 'category_label' in model_forward_params and category_label is not None:
                net_input['category_label'] = category_label
            if 'gra' in model_forward_params and gra is not None:
                net_input['gra'] = gra
            if 'text' in model_forward_params and phrase is not None:
                net_input['text'] = phrase

            return self.net(**net_input)['instances']
        except Exception as e:
            # 如果任何方式都失败，回退到基本调用
            print(f"Warning: Error calling model with specific parameters: {e}")
            call_kwargs = {'image': image_nd, 'points': points_nd}
            sig = inspect.signature(self.net.forward)
            valid_kwargs = {k: v for k, v in [('gra', gra), ('text', phrase), ('category_label', category_label)] if
                            k in sig.parameters}
            call_kwargs.update(valid_kwargs)
            return self.net(**call_kwargs)['instances']

    def apply_transforms(self, image_nd, clicks_lists):
        """应用图像变换"""
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        """将点击转换为模型输入格式"""
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_sam_batched_input(self, image_nd, points_nd):
        batched_output = []
        for i in range(image_nd.shape[0]):
            image = image_nd[i]
            point_length = points_nd[i].shape[0] // 2
            point_coords = []
            point_labels = []
            for i, point in enumerate(points_nd[i]):
                point_np = point.cpu().numpy()
                if point_np[0] == -1:
                    continue
                if i < point_length:
                    point_labels.append(1)
                else:
                    point_labels.append(0)
                point_coords.append([point_np[1], point_np[0]])

            res = {
                'image': image[:3, :, :],
                'point_coords': torch.as_tensor(np.array(point_coords), dtype=torch.float, device=self.device)[None, :],
                'point_labels': torch.as_tensor(np.array(point_labels), dtype=torch.float, device=self.device)[None, :],
                'original_size': image.cpu().numpy().shape[1:],
                'mask_inputs': image[3, :, :][None, None, :] if image.shape[0] > 3 else None
            }
            batched_output.append(res)
        return batched_output

