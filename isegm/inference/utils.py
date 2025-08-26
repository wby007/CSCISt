from datetime import timedelta
from pathlib import Path
import torch
import numpy as np

from isegm.data.datasets import *
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, eval_ritm, lora_checkpoint=None, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, eval_ritm, **kwargs)
        models = [load_single_is_model(x, device, eval_ritm, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, eval_ritm, lora_checkpoint=lora_checkpoint, **kwargs)


def load_single_is_model(state_dict, device, eval_ritm, lora_checkpoint=None, **kwargs):
    _config = state_dict['config']
    if lora_checkpoint is not None:
        lora_state_dict = torch.load(lora_checkpoint, map_location='cpu')
        _config = lora_state_dict['config']

    if hasattr(state_dict, '_config'):
        _config = state_dict._config
    elif isinstance(state_dict, dict) and 'config' in state_dict:
        _config = state_dict['config']
    elif isinstance(state_dict, dict) and 'model_config' in state_dict:
        _config = state_dict['model_config']
    else:
        # 如果没有配置信息，使用默认模型
        from isegm.model.is_plainvit_model import PlainVitModel
        _config = {
            'class': 'isegm.model.is_plainvit_model.PlainVitModel',
            'params': {}
        }

        # 确保_config['class']是字符串而不是函数
    if not isinstance(_config['class'], str):
        raise ValueError(f"Expected config['class'] to be a string, got {type(_config['class'])}")

    model = load_model(_config, eval_ritm, **kwargs)
    msg = model.load_state_dict(state_dict['state_dict'], strict=False)

    if lora_checkpoint is not None:
        msg = model.load_state_dict(lora_state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg, args):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'PascalPart':
        dataset = PascalPartEvaluationDataset(cfg.PASCALPART_PATH, split='val', class_name=args.class_name,
                                              part_name=args.part_name)
    elif dataset_name == 'PartImageNet':
        dataset = PartINEvaluationDataset(cfg.PARTIMAGENET_PATH, split='val', class_name=args.class_name,
                                          part_name=args.part_name)
    elif dataset_name == 'SA1B':
        dataset = SA1BDataset(cfg.SA1B_PATH)
    elif dataset_name == 'BraTS':
        dataset = BraTSDataset(cfg.BraTS_PATH)
    elif dataset_name == 'ssTEM':
        dataset = ssTEMDataset(cfg.ssTEM_PATH)
    elif dataset_name == 'OAIZIB':
        dataset = OAIZIBDataset(cfg.OAIZIB_PATH)
    elif dataset_name == 'COCO2014':
        dataset = COCO2014Dataset(cfg.COCO2014_PATH)
    elif dataset_name == 'COCO2017':
        dataset = COCO2017Dataset(cfg.COCO2017_PATH, split='val')
    elif dataset_name == 'CocoLvis':
        dataset = CocoLvisDataset(cfg.COCO_LVIS_PATH, split='train')  # 假设使用训练集
    elif dataset_name == 'COCOMVal':
        dataset = COCOMValDataset(cfg.COCOMVal_PATH)  # 假设使用训练集
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int_)

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)



from datetime import timedelta
import openpyxl
from openpyxl import Workbook
import os
import datetime


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time, iou_first,
                      n_clicks=20, model_name=None):
    # Define the table header with IoU column
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{"IoU@1":^9}|'
                    f'{">=" + str(n_clicks) + "@85%":^9}|{">=" + str(n_clicks) + "@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time(s)":^9}')  # 修正了标题

    # Calculate the row width based on the header length
    row_width = len(table_header)

    # Add header info with model name
    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    # Format the elapsed time
    eval_time = str(timedelta(seconds=int(elapsed_time)))

    # Prepare the table row with necessary checks on list lengths
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'

    # Handle the noc_list (No of Clicks) values
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'

    # Handle the IoU value (IoU@1)
    table_row += f'{iou_first:^9.2f}|'

    # Handle the over_max_list values
    table_row += f'{over_max_list[1]:^9}|' if len(over_max_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(over_max_list) > 2 else f'{"?":^9}|'

    # Handle mean_spc and eval_time
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    # 获取当前时间
    time_write = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Ensure the directory exists
    directory = 'outputs/experiments'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 保存数据到Excel文件
    data = [[model_name, brs_type, dataset_name, noc_list[0],
             noc_list[1], noc_list[2], iou_first,over_max_list[1], over_max_list[2], mean_spc, elapsed_time, time_write]]

    file_path = os.path.join(directory, 'eval_output_CSAtt_sbd_large.xlsx')
    # ==========================⬛⬛⬛⬛⬛⬛评估结果excel保存⬛⬛⬛⬛⬛⬛==========================
    if os.path.exists(file_path):
        # 文件已存在，打开文件
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
    else:
        # 文件不存在，创建新的工作簿和工作表
        wb = Workbook()
        ws = wb.active
        # 设置标题
        ws.append(
            ["Epoch", "BRS Type", "Dataset", "NoC@80%", "NoC@85%", "NoC@90%", "IoU@1", ">=20@85%", ">=20@90%", "SPC,s",
             "Time(s)", "Time Write"])

    # 将数据写入到excel
    for row in data:
        ws.append(row)

    # 保存Excel文件
    wb.save(file_path)

    return header, table_row

