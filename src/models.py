import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import timm
import segmentation_models_pytorch as smp

class TimmSegModel(nn.Module):
    def __init__(self, backbone, out_dim, segtype='unet',
                 n_blocks=4, drop_rate=0, drop_path_rate=0, pretrained=False
                 ):
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        self.n_blocks = n_blocks
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            # self.decoder = smp.unet.decoder.UnetDecoder(
            #     encoder_channels=encoder_channels[:n_blocks+1],
            #     decoder_channels=decoder_channels[:n_blocks],
            #     n_blocks=n_blocks,
            # )
            self.decoder = smp.Unet()

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


class DASegModel(nn.Module):
    def __init__(self, backbone, in_chans, out_chans, out_chans_domain):
        """
        seg_model: segmentation_models_pytorch のセグメンテーションモデル
        feature_dim: ドメイン識別器に入力する特徴量のチャネル数
        """
        super().__init__()
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        self.seg_model = smp.Unet(encoder_name=backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    in_channels=in_chans,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=out_chans,        # model output channels (number of classes in your dataset)
                    activation=None,
                    aux_params=aux_params
                    )
        in_features = self.seg_model.classification_head[3].in_features
        self.seg_model.classification_head = nn.Identity()
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, out_chans_domain)
        )
        
    def forward(self, x, constant):
        # セグメンテーションモデルの出力
        # segmentation_models_pytorch の場合、メイン出力と auxiliary 出力の2種類を返す設定にできる場合があります
        encoded = self.seg_model.encoder(x)
        features = F.adaptive_avg_pool2d(encoded[-1], (1,1))  # [batch, channels, 1, 1]
        features = features.view(features.size(0), features.size(1))         # [batch, channels]
        out = self.seg_model.decoder(*encoded)
        out = self.seg_model.segmentation_head(out)
        # GRL を適用して、逆伝播で勾配の符号を反転させる
        features_reversed = GradientReversalLayer.apply(features, constant)
        # ドメイン識別器でドメイン判別を行う
        domain_pred = self.domain_classifier(features_reversed).squeeze(1)
        
        return out, domain_pred

class GradientReversalLayer(Function):
    """
    Gradient Reversal Layer (GRL) for domain adaptation.

    This custom autograd function implements a gradient reversal layer, which
    reverses the gradients during backpropagation. It is used in domain
    adversarial training to align feature distributions between domains.

    Methods:
    -------
    forward(context, x, constant): Passes the input through during the forward pass.
    backward(context, grad): Reverses the gradients during the backward pass.

    Examples:
    --------
    >>> grl = GradientReversalLayer.apply
    >>> x = torch.randn(32, 100)
    >>> constant = 1.0
    >>> y = grl(x, constant)
    >>> print(y)
    """
    
    @staticmethod
    def forward(context, x, constant):
        """
        Forward pass of the GRL.

        Parameters:
        ----------
        context: Autograd context object to save information for backward computation.
        x (torch.Tensor): Input tensor.
        constant (float): The constant factor to scale the input.

        Returns:
        -------
        torch.Tensor: The scaled input tensor.
        """
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        """
        Backward pass of the GRL. Reverses the gradients.

        Parameters:
        ----------
        context: Autograd context object to retrieve saved information from the forward pass.
        grad (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
        torch.Tensor: Reversed gradients scaled by the constant factor.
        None: None for the second return value as constant is not a tensor.
        """
        return grad.neg() * context.constant, None


#------------------------------------------------
#2d decoder
class MyDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class MyUnetDecoder(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
            print(block.conv1[0])
            print('')
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode
#------------------------------------------------
#3d decoder
class MyDecoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        #print(in_channel , skip_channel, out_channel,)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None, depth_scaling=2):
        x = F.interpolate(x, scale_factor=(depth_scaling,2,2), mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class MyUnetDecoder3d(nn.Module):
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip, depth_scaling=[2,2,2,2,2,2]):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            # print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
            # print(block.conv1[0])
            # print('')

            s = skip[i]
            d = block(d, s, depth_scaling[i])
            decode.append(d)
        last = d
        return last, decode

def encode_for_resnet(e, x, B, depth_scaling=[2,2,2,2,1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode=[]
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)
    #print(x.shape)
    #x = e.maxpool(x)
    x = F.avg_pool2d(x,kernel_size=2,stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)
    #print(x.shape)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)
    #print(x.shape)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)
    #print(x.shape)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)
    #print(x.shape)

    return encode


class Net(nn.Module):
    def __init__(self, backbone, kernel, padding, pretrained=False):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        num_class=6+1

        self.arch = backbone
        
        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(self.arch, [768])
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1]+[0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1],num_class, kernel_size=kernel, padding=padding)

    def forward(self, image):
        device = self.D.device

        image = image.to(device)
        B, D, H, W = image.shape
        image = image.reshape(B*D, 1, H, W)

        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        #encode = self.encoder(x)[-5:]
        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])

        #[print(f'encode_{i}', e.shape) for i, e in enumerate(encode)]
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=[1,2,2,2,2]
        )
        #print(f'last', last.shape)

        logit = self.mask(last)
        #print('logit', logit.shape)

        return logit


class Net3Axis(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        num_class=6+1

        self.arch = backbone
        
        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(self.arch, [768])
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1]+[0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1],num_class, kernel_size=1)
    

    def forward(self, image):
        device = self.D.device

        image = image.to(device)
        B, D, H, W = image.shape
        image = image.reshape(B*D, 1, H, W)

        x = (image.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        #encode = self.encoder(x)[-5:]
        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])

        #[print(f'encode_{i}', e.shape) for i, e in enumerate(encode)]
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=[1,2,2,2,2]
        )
        #print(f'last', last.shape)

        logit = self.mask(last)
        #print('logit', logit.shape)

        return logit