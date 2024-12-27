import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn.initializer import KaimingNormal, Constant

kaiming_normal_init = KaimingNormal()
constant_init_zero = Constant(0.0)
constant_init_one = Constant(1.0)
import numpy as np
from backbones.mobilenetv3_four_channels import MobileNetV3_1_0


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init_zero(m.bias)

        elif isinstance(m, nn.Conv1D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init_zero(m.bias)

        elif isinstance(m, (nn.BatchNorm2D, nn.InstanceNorm2D)):
            constant_init_one(m.weight)
            if m.bias is not None:
                constant_init_zero(m.bias)
        elif isinstance(m, nn.Linear):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init_zero(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2D):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2D):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.MaxPool2D):
            pass
        elif isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.Hardswish):
            pass
        elif isinstance(m, nn.Hardsigmoid):
            pass
        elif isinstance(m, nn.Upsample):
            pass
        else:
            m.init_weight()


class Network(nn.Layer):
    def __init__(self):
        super(Network, self).__init__()
        self.backbone = MobileNetV3_1_0()
        channels = [24, 40, 112, 160]
        self.squeeze = Squeeze(channels, 32)
        self.modal_fusion = modal_fusion([32] * 4)
        self.decoder_fused = Decoder([32] * 4)
        channels = [32] * 4
        self.linear1 = nn.Conv2D(channels[0], 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2D(channels[1], 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2D(channels[2], 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2D(channels[3], 1, kernel_size=3, stride=1, padding=1)
        self.init_weight()
        for p in self.backbone.parameters():
            p.optimize_attr['learning_rate'] /= 1000.0

    def forward(self, rgb):
        fused = paddle.concat([rgb, rgb[:, 0:1, :, :]], 1)
        size = fused.shape[2:]
        feat1, feat2, feat3, feat4 = self.backbone(fused)
        feat1, feat2, feat3, feat4 = self.squeeze(feat1, feat2, feat3, feat4)
        feat1, feat2, feat3, feat4 = self.modal_fusion(feat1, feat2, feat3, feat4)
        feat1, feat2, feat3, feat4 = self.decoder_fused(feat1, feat2, feat3, feat4)
        feat1 = F.interpolate(self.linear1(feat1), size=size, mode='bilinear')
        feat2 = F.interpolate(self.linear2(feat2), size=size, mode='bilinear')
        feat3 = F.interpolate(self.linear3(feat3), size=size, mode='bilinear')
        feat4 = F.interpolate(self.linear4(feat4), size=size, mode='bilinear')
        return feat1, feat2, feat3, feat4

    def init_weight(self):
        weight_init(self)


class Squeeze(nn.Layer):
    def __init__(self, channels, comm):
        super(Squeeze, self).__init__()
        self.conv1 = ConvBNReLU(channels[0], comm, 3, 1, 1)
        self.conv2 = ConvBNReLU(channels[1], comm, 3, 1, 1)
        self.conv3 = ConvBNReLU(channels[2], comm, 3, 1, 1)
        self.conv4 = ConvBNReLU(channels[3], comm, 3, 1, 1)

    def forward(self, f1, f2, f3, f4):
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f4 = self.conv4(f4)
        return f1, f2, f3, f4

    def init_weight(self):
        weight_init(self)


class ConvBNReLU(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same',
                 dilation=1, groups=1, bias_attr=True, if_relu=True):
        super(ConvBNReLU, self).__init__()
        self.if_relu = if_relu
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding, dilation=dilation, groups=groups,
                                 bias_attr=bias_attr)

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.Hardswish()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.if_relu:
            x = self.relu(x)
        return x

    def init_weight(self):
        weight_init(self)


class modal_fusion(nn.Layer):
    def __init__(self, channels):
        super(modal_fusion, self).__init__()
        self.mf1 = ModalFusion(channels[0], 'low', H=64, W=64)
        self.mf2 = ModalFusion(channels[1], 'low', H=32, W=32)
        self.mf3 = ModalFusion(channels[2], 'high', H=16, W=16)
        self.mf4 = ModalFusion(channels[3], 'high', H=8, W=8)

    def forward(self, feat1, feat2, feat3, feat4):
        feat1 = self.mf1(feat1)
        feat2 = self.mf2(feat2)
        feat3 = self.mf3(feat3)
        feat4 = self.mf4(feat4)
        return feat1, feat2, feat3, feat4

    def init_weight(self):
        weight_init(self)


class MHSA(nn.Layer):
    def __init__(self, n_dims, width=12, height=12, heads=8):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = ConvBNReLU(n_dims, n_dims, 1)
        self.key = ConvBNReLU(n_dims, n_dims, kernel_size=1)
        self.value = ConvBNReLU(n_dims, n_dims, kernel_size=1)

        self.wgt = paddle.create_parameter(
            shape=[1, n_dims, width, height], dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(np.zeros([1, n_dims, width, height]))
        )
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        n_batch, C, width, height = x.shape
        q = self.query(x).reshape((n_batch, self.heads, C // self.heads, -1))
        k = self.key(x).reshape((n_batch, self.heads, C // self.heads, -1))
        v = self.value(x).reshape((n_batch, self.heads, C // self.heads, -1))

        content_content = paddle.matmul(q.transpose((0, 1, 3, 2)), k)
        energy = content_content
        attention = self.softmax(energy)

        out = paddle.matmul(v, attention.transpose((0, 1, 3, 2)))
        out = out.reshape((n_batch, C, width, height))
        out = out * self.wgt + out
        return out

    def init_weight(self):
        weight_init(self)


class ASPP(nn.Layer):
    def __init__(self, in_channel, H, W):
        super(ASPP, self).__init__()
        down_channel = in_channel
        self.conv0 = ConvBNReLU(in_channel, in_channel * 5, 1)
        self.conv1 = nn.Sequential(ConvBNReLU(in_channel, down_channel, 3, 1, padding='same', dilation=1),
                                   AFF(down_channel, 1))
        self.conv2 = nn.Sequential(ConvBNReLU(in_channel, down_channel, 3, 1, padding='same', dilation=2),
                                   AFF(down_channel, 1))
        self.conv3 = nn.Sequential(ConvBNReLU(in_channel, down_channel, 3, 1, padding='same', dilation=4),
                                   AFF(down_channel, 1))
        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
        )
        self.conv_avg = ConvBNReLU(in_channel, down_channel, 1)
        self.att = AFF(down_channel * 4, 2)
        self.conv5 = ConvBNReLU(down_channel * 4, in_channel, 1, 1, 0, if_relu=False)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x5 = self.conv_avg(F.interpolate(self.avg(x), size=x.shape[2:], mode='bilinear'))
        x = self.conv5(self.att(paddle.concat([x1, x2, x3, x5], axis=1)))
        x = F.hardswish(x + identity)
        return x

    def init_weight(self):
        weight_init(self)


class Texture_Enhance(nn.Layer):
    def __init__(self, num_features, H=48, W=48):
        super().__init__()
        self.cv3 = ConvBNReLU(num_features, num_features, 3, 1, 1)
        self.cv4 = ConvBNReLU(num_features, num_features, 3, 1, 1)
        self.cv5 = ConvBNReLU(num_features, num_features, 3, 1, 1, if_relu=False)
        self.cv6 = ConvBNReLU(num_features, num_features, 3, 1, 1, if_relu=False)
        self.att1 = AFF(num_features)

    def forward(self, feature_maps):
        details = F.avg_pool2d(feature_maps, kernel_size=7)
        details = F.interpolate(details, size=feature_maps.shape[2:], mode='bilinear')
        details = paddle.abs(feature_maps - details)
        enhanced_details = feature_maps + details
        enhanced_details1 = self.cv4(enhanced_details)

        enhanced_details2 = self.cv3(feature_maps)
        feature_refined = F.relu(self.cv5(enhanced_details1) + self.cv6(enhanced_details2))
        feature_refined = self.att1(feature_refined)
        return feature_refined

    def init_weight(self):
        weight_init(self)


class ModalFusion(nn.Layer):
    def __init__(self, in_channel, level='high', H=12, W=12):
        super(ModalFusion, self).__init__()
        self.expand = ConvBNReLU(in_channel, in_channel * 2, 1)
        self.cv1 = ConvBNReLU(in_channel, in_channel, 3, 1, 1)
        self.cv2 = ConvBNReLU(in_channel, in_channel, 3, 1, 1)
        self.cv3 = ConvBNReLU(in_channel, in_channel, 3, 1, 1)
        self.aff1 = AFF(in_channel)
        self.aff2 = AFF(in_channel)
        self.aspp = ASPP(in_channel, H=H, W=W) if level == 'high' else Texture_Enhance(in_channel)

    def forward(self, fused):
        fused = self.aspp(fused)
        fused = self.cv3(fused)
        return fused

    def init_weight(self):
        weight_init(self)


class AFF(nn.Layer):
    def __init__(self, channels, r=2):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            ConvBNReLU(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(inter_channels, channels, kernel_size=1, stride=1, padding=0, if_relu=False),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(inter_channels),
            nn.ReLU(),
            nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(channels),
        )

        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        xa = x
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        x = x * wei
        return x

    def init_weight(self):
        weight_init(self)



class RFM(nn.Layer):
    def __init__(self, channels1, channels2, channels3):
        super(RFM, self).__init__()
        self.cv1 = ConvBNReLU(channels1, channels1, 3, 1, 1)
        self.cv2 = ConvBNReLU(channels2, channels1, 3, 1, 1)
        self.cv3 = ConvBNReLU(channels3, 1, 3, 1, 1, if_relu=False)
        self.aff = AFF(channels1 * 2)
        self.mhsa = MHSA(channels1, 8, 8)
        self.cv4 = ConvBNReLU(channels1 * 2, channels1, 3, 1, 1)
        self.cv5 = ConvBNReLU(channels1, channels1, 3, 1, 1)

    def forward(self, left, mid, right):
        left = self.cv1(left)
        mid = F.interpolate(mid, size=left.shape[2:], mode='bilinear')
        mid = self.cv2(mid)
        right = F.interpolate(self.mhsa(right), size=left.shape[2:], mode='bilinear')
        right = F.hardsigmoid(self.cv3(right))

        lr = left * right + left
        mr = mid * right + mid

        cat = paddle.concat([lr, mr], 1)
        cat = self.cv5(self.cv4(self.aff(cat)))
        return cat

    def init_weight(self):
        weight_init(self)


class Decoder(nn.Layer):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.rf1 = RFM(channels[2], channels[3], channels[3])
        self.rf2 = RFM(channels[1], channels[2], channels[3])
        self.rf3 = RFM(channels[0], channels[1], channels[3])

    def forward(self, feat1, feat2, feat3, feat4):
        feat3 = self.rf1(feat3, feat4, feat4)
        feat2 = self.rf2(feat2, feat3, feat4)
        feat1 = self.rf3(feat1, feat2, feat4)
        return feat1, feat2, feat3, feat4

    def init_weight(self):
        weight_init(self)
