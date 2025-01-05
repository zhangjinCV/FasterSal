import paddle
import paddle.nn.functional as F
import sys
import numpy as np
import argparse
import os
import cv2

from net import Network
from lib.dataset import test_dataset


def config():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--dataset_path', default="work/RGB-DSOD/", help='weigted', type=str)
    parser.add_argument("--test_size", default=256, type=int)
    parser.add_argument('--weight', default="weight/TRSENet_RGBD/model-200.pdparams", help='weigted', type=str)
    parser.add_argument('--save_path', default='maps/TRSENet_RGBD/', type=str)
    cag = parser.parse_args()
    return cag


cag = config()
print(cag.weight)
dataset_path = cag.dataset_path
model = Network()
model.load_dict(paddle.load(cag.weight))
model.eval()

# test
test_mae = []
test_datasets = ['NJUD', 'SSD', 'LFSD', 'NLPR', 'SIP', 'DUT-RGBD-Test', 'STERE']
for dataset in test_datasets:
    mae_sum = 0
    save_path = cag.save_path + '/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test_images/'
    gt_root = dataset_path + dataset + '/test_masks/'
    ti_root = dataset_path + dataset + '/test_depth/'
    test_loader = test_dataset(image_root, gt_root, ti_root, cag.test_size)
    for i in range(test_loader.size):
        image, gt, ti, name = test_loader.load_data()
        res = model(image, ti)
        predict = F.sigmoid(res[0])
        predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
        mae = paddle.sum(paddle.abs(predict - gt)) / paddle.numel(gt)
        # mae = torch.abs(predict - gt).mean()
        mae_sum = mae.item() + mae_sum
        predict = predict.numpy().squeeze()
        # print(predict.shape)
        cv2.imwrite(save_path + name, predict * 255)
    test_mae.append(mae_sum / test_loader.size)

for i in range(len(test_mae)):
    print(test_datasets[i], ': ', test_mae[i])
