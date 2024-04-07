# coding: utf-8

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
from sklearn import metrics
from common import constants


def main():

    # ----------------获取参数设置----------------
    batch_size = 4
    epochs = 1
    learning_rate = 5e-6

    device = 'mps'

    # ----------------数据读取与封装----------------
    # 获取到dataset
    train_dataset = CNewsDataset('data/cnews.train_debug.txt')
    valid_dataset = CNewsDataset('data/cnews.val_debug.txt')

    # 封装DataLoader，用于生成Batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # ----------------模型加载----------------
    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(constants.BERT_PATH)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).to(device)

    # ----------------优化器与损失函数配置----------------
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    # ----------------模型训练过程----------------
    for epoch in range(1, epochs+1):
        losses = 0
        accuracy = 0

        model.train()
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()  [4, 10]
            output = model(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                token_type_ids=token_type_ids.to(device), 
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            # output [0.1, 0.05, ..., 0.23]
            pred_labels = torch.argmax(output, dim=1)
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # ----------------模型验证过程----------------
        model.eval()
        losses = 0
        pred_labels = []
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device), 
                attention_mask=attention_mask.to(device), 
                token_type_ids=token_type_ids.to(device), 
            )
            
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)
            acc = torch.sum(pred_label == label_id.to(device)).item() / len(pred_label)
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss = losses / len(valid_dataloader)
        print('\tLoss:', average_loss)
        
        # 分类报告
        report = metrics.classification_report(true_labels, pred_labels, labels=valid_dataset.labels_id, target_names=valid_dataset.labels)
        print('* Classification Report:')
        print(report)

        # f1 用来判断最优模型
        f1 = metrics.f1_score(true_labels, pred_labels, labels=valid_dataset.labels_id, average='micro')
        
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # 判断并保存验证集上表现最好的模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'models/best_model.pkl')


if __name__ == '__main__':
    main()
