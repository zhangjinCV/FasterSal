import datetime
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from lib.dataset import get_loader
import numpy as np
import cv2
import argparse
import os
import random
import paddle.distributed as dist
from visualdl import LogWriter
from scipy.ndimage import distance_transform_edt

# config
def config():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--Min_LR', default=1e-6, help='min lr', type=float)
    parser.add_argument('--Max_LR', default=1e-4, type=float)
    parser.add_argument('--top_epoch', default=20, type=int)
    parser.add_argument('--epoch', default=400, type=int)
    parser.add_argument('--train_bs', default=32, type=int)
    parser.add_argument('--decay', default=5e-4)
    parser.add_argument('--train_size', default=256, type=int)
    parser.add_argument('--momen', default=0.9)
    parser.add_argument('--max_mae', default=1, type=float)
    parser.add_argument('--show_step', default=3, type=int)
    parser.add_argument('--datapath', default=r'work/RGB-DSOD/RGBD_Train')
    parser.add_argument('--test_path', default=r'work/RGB-DSOD/NJUD')
    parser.add_argument('--savepath', default='weight/TRSENet_RGBD')
    parser.add_argument('--save_iter', default=1, help=r'every iter to save model')
    cag = parser.parse_args()
    return cag


cag = config()


# lr scheduler
def lr_decay(steps, scheduler):
    mum_step = cag.top_epoch * global_loader
    min_lr = cag.Min_LR
    max_lr = cag.Max_LR
    total_steps = cag.epoch * global_loader
    if steps < mum_step:
        lr = min_lr + abs(max_lr - min_lr) / (mum_step) * steps
    else:
        lr = scheduler.get_lr()
        scheduler.step()
    return lr



# dice loss
def wiou_loss(pred, mask, weight):
    pred = F.sigmoid(pred)
    sizes = paddle.sum(mask, axis=(1, 2, 3), keepdim=False)
    sizes = sizes / paddle.max(sizes)
    sizes = 1 / sizes
    weight = weight * sizes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    inter = (pred * mask * weight).sum(axis=(2, 3))
    union = ((pred + mask) * weight).sum(axis=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    wiou = wiou.mean()
    return wiou


# train
def train(Network):
    # dataset
    loader = get_loader("work/RGB-DSOD/RGBD_Train/train_images",
                        "work/RGB-DSOD/RGBD_Train/train_masks",
                        "work/RGB-DSOD/RGBD_Train/train_depth",
                        cag.train_bs,
                        cag.train_size
                        )
    dist.init_parallel_env()
    # network
    net = Network()
    net = paddle.DataParallel(net)
    net.train()
    # params
    total_params = sum(p.numel() for p in net.parameters())
    print('total params : ', total_params)

    # optimizer
    clip = paddle.nn.ClipGradByValue(min=-0.5, max=0.5)
    optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=cag.Min_LR, grad_clip=clip)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cag.Max_LR,
                                                         T_max=len(loader) * (cag.epoch - cag.top_epoch),
                                                         eta_min=cag.Min_LR)
    global_step = 0
    global global_loader
    global_loader = len(loader)
    print(global_loader)
    # training
    for epoch in range(0, cag.epoch):
        start = datetime.datetime.now()
        for batch_idx, (image, mask, depth, weight) in enumerate(loader, start=1):
            lr = lr_decay(global_step, scheduler)
            optimizer.clear_grad()
            optimizer.set_lr(lr)

            global_step += 1
            feat1, feat2, feat3, feat4 = net(image, depth)
            loss0 = wiou_loss(feat1, mask, weight)
            loss1 = wiou_loss(feat2, mask, weight)
            loss2 = wiou_loss(feat3, mask, weight)
            loss3 = wiou_loss(feat4, mask, weight)
            loss = loss0 + loss1 / 2 + loss2 / 4 + loss3 / 8
            loss.backward()
            optimizer.step()

            # output log
            if batch_idx % cag.show_step == 0:
                msg = '%s | step:%d/%d/%d (%.2f%%) | lr=%.4f |  loss=%.4f | loss0=%.4f | loss1=%.4f | loss2=%.4f | loss3=%.4f | %s ' % (
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx, epoch + 1, cag.epoch,
                batch_idx / global_loader * 100, optimizer.get_lr(), loss.item(), loss0.item(), loss1.item(),
                loss2.item(), loss3.item(), image.shape)
                print(msg)

        # save weight
        if epoch > cag.epoch / 40 * 39:
            paddle.save(net.state_dict(), cag.savepath + '/model-' + str(epoch + 1) + '.pdparams')

        if epoch % 100 == 0:
            mae = eval(net, cag.test_path)
            print("%.4f" % mae, "%.4f" % cag.max_mae)
            if mae < cag.max_mae:
                cag.max_mae = mae
                paddle.save(net.state_dict(), cag.savepath + '/best_model.pdparams')

        # ETA
        end = datetime.datetime.now()
        spend = int((end - start).seconds)
        eta = datetime.timedelta(seconds=spend * (cag.epoch - epoch))
        eta = datetime.datetime.now() + eta
        mins = spend // 60
        secon = spend % 60
        print(f'this epoch spend {mins} m {secon} s, eta: {eta.strftime("%Y-%m-%d %H:%M:%S")}. \n')


def eval(Network, test_path):
    model = Network
    model.eval()
    from lib.dataset import test_dataset
    mae_sum = []
    image_root = test_path + '/test_images/'
    gt_root = test_path + '/test_masks/'
    ti_root = test_path + '/test_depth/'
    test_loader = test_dataset(image_root, gt_root, ti_root, cag.train_size)
    with paddle.no_grad():
        for i in range(test_loader.size):
            image, gt, ti, name = test_loader.load_data()
            res = model(image, ti)
            predict = F.sigmoid(res[0])
            mae = paddle.mean(paddle.abs(predict - gt))
            mae_sum.append(mae.item())
    model.train()
    return np.mean(mae_sum)


if __name__ == '__main__':
    from net import Network
    train(Network)
