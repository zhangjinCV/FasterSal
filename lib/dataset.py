import os
from PIL import Image
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms
import random
import numpy as np
from PIL import ImageEnhance
import paddle
import cv2
import glob
import paddle


# several data augumentation strategies
def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth


def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomRotation(image, label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, depth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):
        self.trainsize = trainsize
        self.images = glob.glob(image_root + '/*.jpg')  # + glob.glob("work/MSRA10K/image/*.jpg")
        self.gts = glob.glob(gt_root + '/*.png')  # + glob.glob("work/MSRA10K/mask/*.png")
        self.depths = glob.glob(depth_root + '/*.png')  # + glob.glob("work/MSRA10K/depth/*.png")

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])

        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        # gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        depth = paddle.concat([depth, depth, depth], 0)
        weight = self.get_weight_mask(gt)
        return image, gt, depth, weight

    def get_weight_mask(self, mask_tensor):
        per_mask = mask_tensor.numpy().squeeze() * 255
        per_mask = per_mask.astype(np.uint8)
        dist = cv2.distanceTransform(per_mask, distanceType=cv2.DIST_L2, maskSize=5) ** 0.9
        tmp = dist[np.where(dist > 0)]
        dist[np.where(dist > 0)] = np.floor(tmp / np.max(tmp) * 255)
        dist = per_mask - dist
        dist = np.clip(dist, 0, 255)
        dist = dist / 255.
        dist = paddle.to_tensor(dist).unsqueeze(0) * 1.0 + 1
        return dist

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        h = self.trainsize
        w = self.trainsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                  Image.NEAREST)

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=4):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    batch_sampler = paddle.io.DistributedBatchSampler(dataset=dataset,
                                                      batch_size=batchsize,
                                                      shuffle=shuffle,
                                                      )
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=4
    )
    return data_loader


def get_test_loader(image_root, gt_root, depth_root, testsize):
    dataset = test_dataset(image_root, gt_root, depth_root, testsize)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4
                             )
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = glob.glob(image_root + '/*.jpg')
        self.gts = glob.glob(gt_root + '/*.png')
        self.depths = glob.glob(depth_root + '/*.png')

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])

        image = self.transform(image).unsqueeze(0)
        gt = self.gt_transform(gt).unsqueeze(0)
        depth = self.depths_transform(depth)
        depth = paddle.concat([depth, depth, depth], 0)
        depth = depth.unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        h = self.testsize
        w = self.testsize
        return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h),
                                                                                                  Image.NEAREST)

    def __len__(self):
        return self.size


if __name__ == '__main__':
    train_loader = get_loader("work/RGB-DSOD/RGBD_Train/train_images", "work/RGB-DSOD/RGBD_Train/train_masks",
                              "work/RGB-DSOD/RGBD_Train/train_depth", 8, 352)
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(next(iter(train_loader))[2].shape)
    print(next(iter(train_loader))[0].min())
    print(next(iter(train_loader))[0].max())
    print(next(iter(train_loader))[1].min())
    print(next(iter(train_loader))[1].max())
    print(next(iter(train_loader))[2].min())
    print(next(iter(train_loader))[2].max())

    # test_loader = get_test_loader("work/RGB-DSOD/RGBD_Train/train_images", "work/RGB-DSOD/RGBD_Train/train_masks", "work/RGB-DSOD/RGBD_Train/train_depth", 352)
    # print(next(iter(test_loader))[0].shape)
    # print(next(iter(test_loader))[1].shape)
    # print(next(iter(test_loader))[2].shape)
    # print(next(iter(test_loader))[0].min())
    # print(next(iter(test_loader))[0].max())
    # print(next(iter(test_loader))[1].min())
    # print(next(iter(test_loader))[1].max())
    # print(next(iter(test_loader))[2].min())
    # print(next(iter(test_loader))[2].max())
