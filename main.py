from torch_geometric.loader import DataLoader
from transformers import BertTokenizer
from transformers import BertModel, AdamW
from torch.optim import Adam
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
import argparse
import torch
import numpy as np
from torch_geometric.transforms import ToUndirected
from dataloader import CombinedDataset
from torch_geometric.data import Data, Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score,average_precision_score, precision_recall_curve
from model import MergeModel
from dataloader import FNNDataset, TextDataset, collate_fn1


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience  # 待多少个epoch来观察验证损失是否有所改善
        self.verbose = verbose  # 如果为True，那么每次验证损失有改进时都会打印一条消息
        self.counter = 0    # 记录已经有多少次没有看到验证损失的改善了
        self.best_score = None  # 记录最好的验证损失对应的分数
        self.early_stop = False  # 是否应该提前停止训练
        self.val_loss_min = np.Inf  # 跟踪目前为止遇到的最低验证损失
        self.delta = delta  # 被认定为改进，验证损失需要减少的最小量
        self.path = path  # 保存检查点（最佳模型权重）的路径

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:     # 如果这是第一次调用或找到新的最佳分数，就更新 best_score 并保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:  # 如果新分数比 best_score 加上 delta 还要差，则增加计数器
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # 当计数器达到 patience 值，设置 early_stop 为真以指示可以停止训练
                self.early_stop = True
        else:     # 如果找到了新的最佳分数，就更新 best_score 和 counter，并保存模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:  # 如果 verbose 为真，打印一条消息显示旧的和新的验证损失值
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss  # 更新 val_loss_min 为新的最低验证损

def train(model, dataloder, device, epoch, tokenizer, optimizer,batch_size):
    num_epochs = epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloder:
            # 获取数据
            graph_data, texts, labels = batch

            # 将数据移动到设备上
            graph_data = graph_data.to(device)
            print(graph_data.shape)

            # 文本数据预处理
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            loss = model(graph_data, input_ids, attention_mask)
            loss = loss.mean()  # 如果 loss 是一个张量，使用 .mean() 将其转换为一个标量值
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 更新损失
            running_loss += loss.item()
            running_loss = running_loss / batch_size

        # 打印每个 epoch 的平均损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloder)}")


def eval(model, dataloder, device, tokenizer):
    model.eval()
    target_all = []
    score_all = []
    with torch.no_grad():
        for batch in dataloder:
            # 获取数据
            graph_data, texts, labels = batch

            # 将数据移动到设备上
            graph_data = graph_data.to(device)

            labels = labels.to(device)

            # 文本数据预处理
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            loss = model(graph_data, input_ids, attention_mask)

            # # 将score转化为一维数组
            score = loss.flatten()
            # print(score.shape)
            # print(score)

            score_all.append(score.cpu().numpy())
            target_all.append(labels.cpu().numpy())

    # 确保所有元素形状一致后再进行拼接
    score_all = np.concatenate(score_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)
    # print(target_all)

    auc = roc_auc_score(target_all, score_all)
    # 计算不同阈值下的精确度召回率
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(target_all, score_all)

    # 选择 Precision 和 Recall 之间的平衡点
    optimal_idx = np.argmax(2 * precision_vals * recall_vals / (precision_vals + recall_vals))
    optimal_threshold = pr_thresholds[optimal_idx]

    # 使用最优阈值来生成预测
    pred = (score_all > optimal_threshold).astype(int)

    precision = precision_score(target_all, pred, average='macro')
    recall = recall_score(target_all, pred, average='macro')
    f1 = f1_score(target_all, pred, average='macro')
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    # print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return auc, precision, recall, f1

def main():
    # 创建一个命令行参数解析器对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # original model parameters
    # 使用add_argument方法添加了各种命令行参数，包括随机种子、设备、超参数等
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    # parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use')
    # hyper-parameters
    # parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
    parser.add_argument('--dataset', type=str, default='gossipcop', help='[politifact, gossipcop]')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--concat', type=bool, default=False, help='whether concat news embedding and graph embedding')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
    parser.add_argument('--feature', type=str, default='spacy', help='feature type, [profile, spacy, bert, content]')
    # 解析命令行参数并将结果存储在args变量中
    args = parser.parse_args()
    # torch.manual_seed(args.seed)  # 设置了PyTorch的随机数生成器的种子，以确保实验的可重复性
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(args.seed)

    # 读取数据集
    df_fake_gossip = pd.read_csv('/root/autodl-tmp/tmp/Code/datasets/FakeNewsData/gossipcop_fake.csv')
    df_real_gossip = pd.read_csv('/root/autodl-tmp/tmp/Code/datasets/FakeNewsData/gossipcop_real.csv')
    df_fake_politifact = pd.read_csv('/root/autodl-tmp/tmp/Code/datasets/FakeNewsData/politifact_fake.csv')
    df_real_politifact = pd.read_csv('/root/autodl-tmp/tmp/Code/datasets/FakeNewsData/politifact_real.csv')
    df_fake_gossip['label'] = 1
    df_real_gossip['label'] = 0
    df_fake_politifact['label'] = 1
    df_real_politifact['label'] = 0


    df = pd.concat([df_fake_gossip, df_real_gossip])
    # df = pd.concat([df_fake_politifact, df_real_politifact])



    # 提取文本和标签
    texts = df['title'].tolist()
    labels = df['label'].tolist()

    # 创建训练集和测试集的实例
    dataset = CombinedDataset(root='/root/autodl-tmp/tmp/Code/datasets/FakeNewsData', name='gossipcop', feature='spacy',
                              texts=texts, labels=labels, empty=False)
    # dataset = CombinedDataset(root='/root/autodl-tmp/tmp/Code/datasets/FakeNewsData', name='politifact', feature='spacy',
    #                           texts=texts, labels=labels, empty=False)

    first_item = dataset[0]

    # 输出图数据、文本和标签
    graph_data, text, label = first_item
    print("Graph Data:", graph_data)
    print("Text:", text)
    print("Label:", label)

    # 获取 train_idx 和 test_idx
    train_idx = dataset.train_idx
    test_idx = dataset.test_idx

    # 创建训练集和测试集
    train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset, test_idx.tolist())

    # 设置批大小和其他参数
    batch_size = 128

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 打印一个批次的训练集数据
    for batch in train_loader:
        graph_data, texts_batch, labels_batch = batch
        print("训练集批次:")
        print("图数据:", graph_data)
        print("文本:", texts_batch)
        print("标签:", labels_batch)
        break  # 只打印一个批次

    # 打印一个批次的测试集数据
    for batch in test_loader:
        graph_data, texts_batch, labels_batch = batch
        print("测试集批次:")
        print("图数据:", graph_data)
        print("文本:", texts_batch)
        print("标签:", labels_batch)
        break

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    gin_config = {"hidden_dim": args.nhid, "num_layers": 2, "device": device,
                  "norm_layer": 0, "aggregation": "mean", "bias": "true"}

    bert_config = BertModel.from_pretrained('bert-base-uncased').config
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = MergeModel(dim_features=300, config1=gin_config,
                       config2=bert_config, device=device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    epoch = args.epochs
    seeds = args.seed
    batch_size = args.batch_size
    auc_list, precision_list, recall_list, f1_list = [], [], [], []
    # 创建一个早停实例
    early_stopping = EarlyStopping(patience=7, verbose=True, path='/root/autodl-tmp/tmp/Code/checkpoint.pt')
    # 训练循环
    for current_epoch in range(epoch):
        print(f"####### Run epoch:{current_epoch + 1} ")

        train(model, train_loader, device, current_epoch + 1, tokenizer, optimizer, batch_size)

        # 在每个 epoch 结束后评估模型
        auc, precision, recall, f1 = eval(model, test_loader, device, tokenizer)

        # 打印评估结果
        print(f"Epoch {current_epoch + 1}/{epoch}, "
              f" AUC: {auc}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

        # 将当前的AUC值作为早停的标准
        early_stopping(-auc, model)  # 注意这里传入的是负AUC，因为我们需要最大化AUC而不是最小化它

        # 如果触发了早停条件，则退出训练循环
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # 加载最优模型权重
        model.load_state_dict(torch.load(early_stopping.path))

        auc_list.append(auc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)


    # 最终评估结果
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)
    precision_mean, precision_std = np.mean(precision_list), np.std(precision_list)
    recall_mean, recall_std = np.mean(recall_list), np.std(recall_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
    print(f"Auc: Mean={auc_mean}, Std={auc_std}")
    print(f"Precision: Mean={precision_mean}, Std={precision_std}")
    print(f"Recall: Mean={recall_mean}, Std={recall_std}")
    print(f"F1 Score: Mean={f1_mean}, Std={f1_std}")

if __name__ == "__main__":
    main()






