import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import matplotlib
from itertools import chain

logger = logging.getLogger('MSA')

class CMCM():
    def __init__(self, args):

        self.args = args
        self.args.tasks = "M"
        self.metrics = MetricsTop(args).getMetics(args.datasetName)

        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def do_train(self, model, dataloader):

        scaler = GradScaler()
        optimizer = optim.AdamW(model.Model.parameters(), lr= self.args.learning_rate, eps=1e-4)
        total_steps = len(dataloader['train'])*self.args.warm_up_epochs   #大致的一个训练step数
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
        #                               threshold=0.0001, eps=1e-08)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

        saved_labels = {}
        # init labels
        logger.info("Init labels...")
        # with tqdm(dataloader['train']) as td:
        #     for batch_data in td:
        #         if self.args.train_mode == 'regression':
        #             labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
        #         else:
        #             labels_m = batch_data['labels']['M']
        #         indexes = batch_data['index'].view(-1)
        #         # self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        losses = []

        CPC_Losses = []
        # valid_F1 = []
        lr = []
        min_or_max = 'min' if self.args.KeyEval in ['MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0     #评价阈值的初始化
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': []}
            y_true = {'M': []}
            model.train()
            train_loss = 0.0
            CPC_Loss_sum = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()      #在训练1个batch之后停止梯度清0，当新的epoch来临时才清0
                    left_epochs -= 1                #这么做相当于把batch_size扩大为（N-1）*batch_size，其中N为一个epoch中的batch数

                    # optimizer.zero_grad()
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if self.args.train_mode == 'regression':
                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        prefix_label = batch_data['labels_prefix']
                        cur_id = batch_data['id']
                        ids.extend(cur_id)
                    else:
                        labels_m = batch_data['labels']['M']

                    indexes = batch_data['index'].view(-1)


                    if not self.args.need_data_aligned:
                        text_lengths = batch_data['text_lengths'].to(self.args.device)
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)

                    # forward
                    with autocast():
                        output= model(labels_m, (text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))
                        loss = output['Loss']



                    # backward
                    scaler.scale(loss).backward()
                    train_loss += loss.item()
                    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
                    # update parameters
                    if not left_epochs:
                        # update
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    scaler.step(optimizer)
                    scaler.update()
            # scheduler.step()   #每个epoch衰减一次学习率
            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f" % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            losses.append(train_loss)

            # validation

            if epochs >= 1:         #前3epochs不做eval
                val_results = self.do_test(model, dataloader['valid'], mode="VAL")
                cur_valid = val_results[self.args.KeyEval]
                # valid_losses.append(val_results['Loss'])
                # valid_F1.append(cur_valid)
                # save best model
                isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                if isBetter:
                    best_valid, best_epoch = cur_valid, epochs
                    # save model
                    # torch.save(model.cpu().state_dict(), self.args.model_save_path)
                    self.save_model(model, epochs, self.args.model_save_path)
                    model.to(self.args.device)

                # # save labels
                # if self.args.save_labels:
                #     tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
                #     tmp_save['ids'] = ids
                #     saved_labels[epochs] = tmp_save
                # early stop
                if epochs - best_epoch >= self.args.early_stop:     #如果比best_epoch再过了early_stop轮之后还没有出现新的best_epoch，就停止训练
                    if self.args.save_labels:
                        with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                            plk.dump(saved_labels, df, protocol=4)
                    # self.loss_plt(losses,CPC_Losses)
                    # self.lr_plt(lr)
                    return


    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        # eval_loss = 0.0
        # criterion = nn.L1Loss()
        if self.args.train_mode == 'regression':
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast():
                            outputs = model.generate((text,text_lengths), (audio, audio_lengths), (vision, vision_lengths))

                        predict_label = torch.Tensor(outputs).to(self.args.device)

                        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
                        # loss = self.l1_loss(predict_label, labels_m)
                        # eval_loss += loss.item()
                        y_pred['M'].append(predict_label.cpu())
                        y_true['M'].append(labels_m.cpu())
            pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
            # print(pred)
            # eval_loss = eval_loss / len(dataloader)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>" )
            eval_results = self.metrics(pred, true)
            logger.info('M: >> ' + dict_to_str(eval_results))
            # eval_results['Loss'] = eval_loss
        else:
            # train_mode == 'classification'
            with torch.no_grad():
                with tqdm(dataloader) as td:
                    for batch_data in td:
                        vision = batch_data['vision'].to(self.args.device)
                        audio = batch_data['audio'].to(self.args.device)
                        text = batch_data['text'].to(self.args.device)
                        if not self.args.need_data_aligned:
                            text_lengths = batch_data['text_lengths'].to(self.args.device)
                            audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                            vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                        with autocast():
                            outputs = model.generate((text, text_lengths), (audio, audio_lengths),
                                                     (vision, vision_lengths))

                        # predict_label = torch.Tensor(outputs).to(self.args.device)
                        predict_label = outputs
                        labels_m = batch_data['labels']['M']
                        # y_pred['M'].append(predict_label.cpu().numpy())
                        y_pred['M'].append(predict_label)
                        y_true['M'].append(labels_m)
            # pred, true = torch.cat(y_pred['M']), torch.cat(y_pred['M'])
            pred, true = list(chain(*y_pred['M'])), list(chain(*y_true['M']))
            # print(pred)
            eval_results = self.metrics(pred, true)
            logger.info(mode + "-(%s)" % self.args.modelName + " >>")
            logger.info('M: >> ' + dict_to_str(eval_results))

        return eval_results
    
    def l1_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            loss = torch.mean(torch.abs(y_pred - y_true))
        return loss



    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def save_model(self, model, epoch, save_path):
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.cpu().state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        logging.info("Saving checkpoint at epoch {} to {}.".format(epoch, save_path))
        torch.save(state_dict, save_path)


    # def loss_plt(self,loss,CPC_Losses):
    #     matplotlib.rcParams['font.family'] = 'serif'  # 设置字体族
    #     matplotlib.rcParams['font.serif'] = ['Arial']  # 选择字体
    #     logging.getLogger('matplotlib').setLevel(logging.ERROR)
    #     # train_x = range(len(loss))
    #     # train_y = loss
    #     # kl_x = range(len(KL_losses))
    #     # kl_y = KL_losses
    #     #
    #     # save_path = os.path.join(self.args.res_save_dir, f'{self.args.datasetName}-{self.args.train_mode}.jpg')
    #     # fig, axs = plt.subplots(2, 1)
    #     #
    #     # # Plot Train Loss
    #     # axs[0].plot(train_x, train_y, label='Train')
    #     # axs[0].set_ylabel('Loss')
    #     # axs[0].set_ylim([0, max(train_y) * 1.2])
    #     # axs[0].set_yticks(np.arange(0, max(train_y) + 0.1, (max(train_y) - min(train_y)) / 5))
    #     # axs[0].legend(loc='upper right')
    #     #
    #     # # Plot KL Loss with Log Scale
    #     # axs[1].plot(kl_x, kl_y, label='KL_losses')
    #     # axs[1].set_yscale('log')  # Set log scale for KL Loss
    #     # axs[1].set_ylabel('KL-Loss')
    #     # axs[1].set_ylim([min(kl_y) - 0.05, max(kl_y) + 0.05])
    #     # axs[1].set_yticks(np.arange(min(kl_y), max(kl_y) + 0.01, (max(kl_y) - min(kl_y)) / 5))
    #     # axs[1].legend(loc='upper right')
    #     #
    #     # plt.xlabel('epoch')
    #     # plt.subplots_adjust(hspace=0.5)
    #     # plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    #     # plt.close()
    #     # plt.show()
    #     kl_x = range(len(CPC_Losses))
    #     kl_y = CPC_Losses
    #
    #     save_path = os.path.join(self.args.res_save_dir, f'{self.args.datasetName}-{self.args.train_mode}_CPC_Losses.jpg')
    #     fig, ax = plt.subplots(figsize=(8, 6))  # 调整图的大小
    #
    #     # Plot KL Loss with Log Scale
    #     ax.plot(kl_x, kl_y, label='KL_losses')
    #     ax.set_yscale('log')  # 设置 KL Loss 的纵坐标为对数坐标
    #     ax.set_ylabel('KL-Loss')
    #     ax.set_ylim([min(kl_y) - 0.05, max(kl_y) + 0.05])
    #
    #     # 使用自动设置刻度
    #     ax.yaxis.set_major_locator(plt.AutoLocator())
    #
    #     ax.legend(loc='upper right')
    # 
    #     plt.xlabel('epoch')
    #     plt.tight_layout()  # 自动调整布局
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    #     plt.close()
    #     plt.show()

    # def lr_plt(self,lr):
    #     plt.plot(np.arange(len(lr)), lr)
    #     plt.xlabel('Step')
    #     plt.ylabel('Learning Rate')
    #     plt.title('Warm-up Learning Rate Schedule')
    #     plt.show()
