import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
from utils.dataloader import test_dataset
import tqdm
from PIL import Image
from tabulate import tabulate
from utils.eval_functions import *
from lib.networks import DBAANet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./model_pth/Polyp/HDA3/99DualHDA3_polyp-best.pth')
parser.add_argument('--data_path', type=str, default='./data/polyp/TestDataset/')
parser.add_argument('--save_path', type=str, default='./predict_score/DBAA/')

opt = parser.parse_args()

def evaluate(pred_path, gt_path, verbose=True):
    result_path = 'results'
    method = 'DBAA'
    Thresholds = np.linspace(1, 0, 256)
    headers = ['dataset', 'meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen', 'meanSpe', 'maxSpe', 'Dice_0.5', 'Jaccard_0.5']
    results = []
    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    
    if verbose:
        print('#' * 20, 'Start Evaluation', '#' * 20)
        datasets = tqdm.tqdm(datasets, desc='Expr - ' + method, total=len(datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    
    for dataset in datasets:
        pred_root = os.path.join(pred_path, dataset)
        gt_root = os.path.join(gt_path, dataset, 'masks')
        preds = os.listdir(pred_root)
        gts = os.listdir(gt_root)
        preds.sort()
        gts.sort()

        threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
        threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))
        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))
        Dice_05 = np.zeros(len(preds))
        Jaccard_05 = np.zeros(len(preds))

        if verbose:
            samples = tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(preds), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))

        for i, sample in samples:
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]
            pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]
            assert pred_mask.shape == gt_mask.shape
            gt_mask = gt_mask.astype(np.float64) / 255
            gt_mask = (gt_mask > 0.5).astype(np.float64)
            pred_mask = pred_mask.astype(np.float64) / 255
            Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
            wFmeasure[i] = original_WFb(pred_mask, gt_mask)
            MAE[i] = np.mean(np.abs(gt_mask - pred_mask))
            
            # 固定阈值 0.5 的 Dice 和 Jaccard
            input = np.where(pred_mask >= 0.5, 1, 0)
            target = np.where(gt_mask >= 0.5, 1, 0)
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection
            Jaccard_05[i] = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
            Dice_05[i] = (2 * np.sum(intersection) + smooth) / (np.sum(input) + np.sum(target) + smooth)
            
            threshold_E = np.zeros(len(Thresholds))
            threshold_F = np.zeros(len(Thresholds))
            threshold_Pr = np.zeros(len(Thresholds))
            threshold_Rec = np.zeros(len(Thresholds))
            threshold_Iou = np.zeros(len(Thresholds))
            threshold_Spe = np.zeros(len(Thresholds))
            threshold_Dic = np.zeros(len(Thresholds))
            for j, threshold in enumerate(Thresholds):
                threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)
                Bi_pred = np.zeros_like(pred_mask)
                Bi_pred[pred_mask >= threshold] = 1
                threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []
        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)
        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)
        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)
        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)
        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)
        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)
        dice_05 = np.mean(Dice_05)
        jaccard_05 = np.mean(Jaccard_05)
        
        result.extend([meanDic, meanIoU, wFm, Sm, meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe, dice_05, jaccard_05])
        results.append([dataset, *result])

        csv = os.path.join(result_path, 'result_' + dataset + '.csv')
        if os.path.isfile(csv):
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join(headers) + '\n')
        out_str = method + ',' + ','.join(['{:.4f}'.format(metric) for metric in result]) + '\n'
        csv.write(out_str)
        csv.close()
    
    tab = tabulate(results, headers=headers, floatfmt=".3f")
    if verbose:
        print(tab)
        print("#"*20, "End Evaluation", "#"*20)
    return tab

if __name__ == "__main__":
    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    
    # 确保 EMCADNet 结构与训练时一致
    model = DBAANet()  # 需更新为与训练时相同的结构
    model.load_state_dict(torch.load(opt.pth_path), strict=True)  # 使用 strict=True 验证
    model.cuda()
    model.eval()

    for _data_name in datasets:
        data_path = os.path.join(opt.data_path, _data_name)
        save_path = os.path.join(opt.save_path, _data_name)
        os.makedirs(save_path, exist_ok=True)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            predicts = model(image, mode='test')  # 显式指定 mode='test'
            res = sum(predict.sigmoid() for predict in predicts)  # 融合四个深监督输出并应用 Sigmoid
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(os.path.join(save_path, name), ((res > 0.5) * 255).astype(np.uint8))
    
    evaluate(opt.save_path, opt.data_path)