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

class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
        # text enocding
        self.LLM = Language_model(args)

        # audio and video enocding
        text_in, audio_in, video_in = args.feature_dims[:]
        text_len, audio_len, video_len = args.seq_lens[:]

        self.audio_LSTM = TVA_LSTM(audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(video_in, args.v_lstm_hidden_size, num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        self.text_guide_mixer = Text_guide_mixer()
        #low_rank_fusion
        fusion_input_size = 256
        self.mutli_scale_fusion = mutli_scale_fusion(input_size=fusion_input_size, output_size= text_in, pseudo_tokens= args.pseudo_tokens)


    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long())

        video_h = self.video_LSTM(video, video_len)
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
        video_h = self.video_LSTM(video, video_len)


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
