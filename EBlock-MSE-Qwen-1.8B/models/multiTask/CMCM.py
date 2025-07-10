# self supervised multimodal multi-task learning network
import math
import os
import sys
import collections
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.Textmodel import Language_model

__all__ = ['CMCM']

# EBlock相关类定义
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Branch(nn.Module):
    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)
        )

    def forward(self, input):
        return self.branch(input)

class EBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1, groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)
        x = y * x_freq
        x = y + x * self.gamma
        return x

class VideoProcessor(nn.Module):
    def __init__(self, video_in):
        super(VideoProcessor, self).__init__()
        # 输入投影层：将视频特征维度投影到合适的通道数
        self.input_proj = nn.Linear(video_in, 64)
        
        # EBlock处理
        self.eblock = EBlock(c=64, DW_Expand=2, dilations=[1, 2, 3])
        
        # 全局平均池化和输出投影
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_proj = nn.Linear(64, 256)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, lengths):
        batch_size, seq_len, feat_dim = x.shape
        
        # 投影到目标通道数
        x = self.input_proj(x)  # [batch, seq, 64]
        
        # 重塑为4D张量用于EBlock处理
        # 将序列长度作为空间维度
        h = w = int(math.sqrt(seq_len))
        if h * w != seq_len:
            # 如果不是完全平方数，进行padding
            target_len = (h + 1) ** 2
            pad_len = target_len - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))
            h = w = h + 1
        
        x = x.permute(0, 2, 1).reshape(batch_size, 64, h, w)  # [batch, 64, h, w]
        
        # EBlock处理
        x = self.eblock(x)
        
        # 全局池化获得最终特征
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # [batch, 64]
        
        # 投影到256维
        x = self.output_proj(x)
        x = self.dropout(x)
        
        return x

class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
        # text enocding
        self.LLM = Language_model(args)

        # audio and video enocding
        text_in, audio_in, video_in = args.feature_dims[:]
        text_len, audio_len, video_len = args.seq_lens[:]

        self.audio_LSTM = TVA_LSTM(audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        # 使用EBlock处理视频
        self.video_processor = VideoProcessor(video_in)

        self.text_guide_mixer = Text_guide_mixer()
        #low_rank_fusion
        fusion_input_size = 256
        self.mutli_scale_fusion = mutli_scale_fusion(input_size=fusion_input_size, output_size= text_in, pseudo_tokens= args.pseudo_tokens)


    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long())

        video_h = self.video_processor(video, video_len)
        audio_h = self.audio_LSTM(audio, audio_len)


        fusion_h= self.text_guide_mixer(audio_h, video_h, text)

        fusion_h= self.mutli_scale_fusion(fusion_h)


        LLM_input = torch.cat([fusion_h, text], dim=1)

        LLM_output = self.LLM(LLM_input, labels)

        res = {
            'Loss': LLM_output.loss,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long())

        audio_h = self.audio_LSTM(audio, audio_len)
        video_h = self.video_processor(video, video_len)


        fusion_h = self.text_guide_mixer(audio_h, video_h, text)

        # low_rank_fusion

        fusion_h = self.mutli_scale_fusion(fusion_h)

        # concatenate mutli_scale_fusion and text_embedding

        LLM_input = torch.cat([fusion_h, text], dim=1)

        LLM_output = self.LLM.generate(LLM_input)

        return LLM_output



class TVA_LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TVA_LSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 256)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False) #这里把length.to cpu是因为pytorch版本问题
        # _, (final_states, _) = self.rnn(packed_sequence)
        # h = self.dropout(final_states[-1])
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        h = self.linear(h)
        return h

class Text_guide_mixer(nn.Module):
    def __init__(self):
        super(Text_guide_mixer, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_mlp = nn.Linear(2048, 256)
    def forward(self, audio, video, text):
        text_GAP = self.GAP(text.permute(0, 2, 1)).squeeze()
        text_knowledge = self.text_mlp(text_GAP)

        audio_mixed = torch.mul(audio, text_knowledge)
        video_mixed = torch.mul(video, text_knowledge)

        fusion = audio_mixed + video_mixed

        return fusion


class mutli_scale_fusion(nn.Module):
    def __init__(self, input_size, output_size, pseudo_tokens = 4):
        super(mutli_scale_fusion, self).__init__()
        multi_scale_hidden = 256
        self.scale1 = nn.Sequential(
            nn.Linear(input_size, output_size // 8),
            nn.GELU(),
            nn.Linear(output_size // 8, multi_scale_hidden)
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_size, output_size // 32),
            nn.GELU(),
            nn.Linear(output_size // 32, multi_scale_hidden)
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_size, output_size // 16),
            nn.GELU(),
            nn.Linear(output_size // 16, multi_scale_hidden)
        )

        self.integrating = Integrating(scales = 3)
        self.multi_scale_projector =  nn.Linear(multi_scale_hidden, output_size)
        self.projector = nn.Linear(1, pseudo_tokens)

    def forward(self,x):
        # 增加样本复制，将单一样本复制一份,避免最后一个batch只有一个数据时的报错
        if x.dim() == 1:
            x = x.unsqueeze(0)
        #compute different scale experts outputs
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)


        # Calculate the expert outputs
        multi_scale_stack = torch.stack([scale1, scale2, scale3], dim=2)
        multi_scale_integrating =  self.integrating(multi_scale_stack)

        multi_scale = self.multi_scale_projector(multi_scale_integrating)
        output = self.projector(multi_scale.unsqueeze(2))
        return output.permute(0, 2, 1)  #[batch,seq_len,hidden_siez]

# Define the gating model
class Integrating(nn.Module):
    def __init__(self,  scales):
        super(Integrating, self).__init__()

    # Layers
        self.Integrating_layer = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, scales), stride=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Integrating_layer(x)
        x = x.squeeze((1, 3))
        return x
