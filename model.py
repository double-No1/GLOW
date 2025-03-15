import torch
import torch.nn as nn
from torch.nn import functional as F
from Code.gin_model.Models import OCGIN
from bert_model import BertEmbeddingModel
from Code.Losses import OCC_loss, InfoNCELoss




class MergeModel(nn.Module):
    def __init__(self, dim_features, config1, config2, device):
        super(MergeModel, self).__init__()

        self.dim_features = dim_features
        self.config1 = config1
        self.config2 = config2
        self.device = device

        self.gin_features = OCGIN(dim_features, config1).to(device)
        # self.bert = BertEmbeddingModel(config2).to(device)
        self.bert = BertEmbeddingModel(config2, output_dim=config1['hidden_dim'] * config1['num_layers']).to(device)
        self.info_nce_loss = InfoNCELoss()
        self.occ_loss = OCC_loss()
        self.w1 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        # self.center = nn.Parameter(torch.empty(1, 1, config1['hidden_dim'] * config1['num_layers']), requires_grad=True)
        self.center = nn.Parameter(torch.empty(1, config1['hidden_dim'] * config1['num_layers']), requires_grad=True)
        # print(self.center.shape)
        nn.init.normal_(self.center, mean=0, std=0.1)
        self.center.data = self.center.data.to(device) 



    def forward(self, data, input_ids, attention_mask):

        # Forward pass through OCGIN
        outputs_1, _ = self.gin_features(data)  # 注意这里返回两个元素，我们需要第一个和第二个
        # print(outputs_1.shape)

        # Forward pass through BERT
        outputs_2 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(outputs_2.shape)

        outputs_1 = F.normalize(outputs_1, dim=1)
        outputs_2 = F.normalize(outputs_2, dim=1)
        # print('output',outputs_1.shape)

        # center
        center = self.center
        # print('center',center.shape)

        # 计算聚合程度（奖励信号）
        cos_sim_1 = F.cosine_similarity(outputs_1, center, dim=1).mean()  # outputs_1 与 center 的相似度
        cos_sim_2 = F.cosine_similarity(outputs_2, center, dim=1).mean()  # outputs_2 与 center 的相似度

        # 奖励信号计算（可根据任务需求调整奖励函数）
        reward_1 = torch.exp(cos_sim_1)  # outputs_1 的奖励
        reward_2 = torch.exp(cos_sim_2)  # outputs_2 的奖励

        # 权重动态调整
        step_size = 0.02  # 动作步长
        weights = F.softmax(torch.tensor([reward_1.item(), reward_2.item()]), dim=0)  # 归一化权重
        w1 = weights[0] + step_size * (reward_1 - reward_2).item()  # 动态调整 outputs_1 的权重
        w2 = weights[1] - step_size * (reward_1 - reward_2).item()  # 动态调整 outputs_2 的权重

        # 确保权重非负并归一化
        w1 = max(0.0, w1)
        w2 = max(0.0, w2)
        weight_sum = w1 + w2
        w1 /= weight_sum
        w2 /= weight_sum



        # Compute InfoNCE loss between outputs_1 and outputs_2
        info_nce_loss = self.info_nce_loss(outputs_1, outputs_2)

        # Compute OCC loss
        # occ_loss = ((self.occ_loss(outputs_1, center) + self.occ_loss(outputs_2, center))/2)
        occ_loss = self.w1*(self.occ_loss(outputs_1, center)) +self.w2*(self.occ_loss(outputs_2, center))

        occ_loss.to(self.device)

        # Combine losses with a weight
        alpha = 0.5  # 这里设置 InfoNCE 和 OCC 损失的权重
        total_loss = alpha * info_nce_loss + (1 - alpha) * occ_loss


        return total_loss

