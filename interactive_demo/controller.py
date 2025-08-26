import torch
import numpy as np
from tkinter import messagebox
import cv2
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
import numpy as np
from isegm.utils.vis import draw_probmap
# 新增
from isegm.utils.vis import visualize_instances, get_boundaries

class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5, granularity=1.0, phrase=None):
        self.net = net
        self.prob_thresh = prob_thresh
        self.granularity = granularity
        self.phrase = phrase
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()

        # 新增
        self.boundaries_color = (121,185,79)  # 边界颜色
        self.boundaries_width = 5
        self.boundaries_alpha = 1

    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        # 保存状态（如果方法存在）
        state_entry = {'clicker': self.clicker.get_state()}
        if hasattr(self.predictor, 'get_states'):
            state_entry['predictor'] = self.predictor.get_states()

        self.states.append(state_entry)
        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)

        # 构建参数字典，只传递非None的参数
        pred_kwargs = {
            'clicker': self.clicker,
            'prev_mask': self._init_mask
        }

        if self.granularity is not None:
            pred_kwargs['gra'] = self.granularity
        if self.phrase is not None:
            pred_kwargs['phrase'] = self.phrase

        pred = self.predictor.get_prediction(**pred_kwargs)

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        # 恢复预测器状态（如果方法存在）
        if hasattr(self.predictor, 'set_states') and 'predictor' in prev_state:
            self.predictor.set_states(prev_state['predictor'])
        # self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask




    '''
    # 新增边界线 方案1  实际采纳
    def get_visualization(self, alpha_blend, click_radius):
        # 参数:
        # alpha_blend: 掩码与原图叠加的透明度
        # click_radius: 显示用户点击标记的半径大小
        # 返回:
        #     包含掩码和交互标记的可视化图像
        if self.image is None:
            return None
        results_mask_for_vis = self.result_mask

        # 只取第一次点击
        first_click = self.clicker.clicks_list[:1] if self.clicker.clicks_list else []

        # 初始化可视化结果，基于原始图像绘制当前掩码和点击标记,参数:当前分割掩码\透明度\点击历史\点击半径
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=first_click, radius=click_radius)
        if self.probs_history:
            mask_vis_b = visualize_instances(
                results_mask_for_vis,  # 实例分割掩码（不同实例用不同整数表示）
                bg_color=None,  # 不覆盖原图背景
                boundaries_color=self.boundaries_color,  # 边界颜色（BGR格式，如蓝色(255,0,0)）
                boundaries_width=self.boundaries_width,  # 边界宽度（像素）
                boundaries_alpha=self.boundaries_alpha  # 边界透明度（0-1）
            )

            # 将带边界的掩码叠加到原图上
            vis = cv2.addWeighted(
                vis,  # 原图或已绘制掩码的图
                0.8,  # 原图权重
                mask_vis_b,  # 带边界的掩码图
                0.6,  # 掩码边界权重
                0  # 亮度调整
            )
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)
            # ✅ 核心边界绘制代码

        return vis
        '''

    '''



    
    #源代码
    def get_visualization(self, alpha_blend, click_radius):
        # 参数:
        # alpha_blend: 掩码与原图叠加的透明度
        # click_radius: 显示用户点击标记的半径大小
        # 返回:
        #     包含掩码和交互标记的可视化图像
        if self.image is None:
            return None
        results_mask_for_vis = self.result_mask

        # 初始化可视化结果，基于原始图像绘制当前掩码和点击标记,参数:当前分割掩码\透明度\点击历史\点击半径
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
        
        
        '''

    def get_visualization(self, alpha_blend, click_radius):
        # 参数:
        # alpha_blend: 掩码与原图叠加的透明度
        # click_radius: 显示用户点击标记的半径大小
        # 返回:
        #     包含掩码和交互标记的可视化图像
        if self.image is None:
            return None
        results_mask_for_vis = self.result_mask

        # 取
        first_two_clicks = self.clicker.clicks_list[:]

        # 初始化可视化结果，基于原始图像绘制当前掩码和点击标记,参数:当前分割掩码\透明度\点击历史\点击半径
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=first_two_clicks, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend,
                                             clicks_list=first_two_clicks, radius=click_radius)

        return vis


    #新增
    def get_heatmap_visualization(self):
        prob_map = self.current_object_prob
        if prob_map is not None:
            heatmap = draw_probmap(prob_map)
            return heatmap
        return None
