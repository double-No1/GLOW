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

        outputs_1, _ = self.gin_features(data)  
        outputs_2 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs_1 = F.normalize(outputs_1, dim=1)
        outputs_2 = F.normalize(outputs_2, dim=1) 
        center = self.center

        cos_sim_1 = F.cosine_similarity(outputs_1, center, dim=1).mean() 
        cos_sim_2 = F.cosine_similarity(outputs_2, center, dim=1).mean()  

       
        reward_1 = torch.exp(cos_sim_1) 
        reward_2 = torch.exp(cos_sim_2) 

       
        step_size = 0.02 
        weights = F.softmax(torch.tensor([reward_1.item(), reward_2.item()]), dim=0) 
        w1 = weights[0] + step_size * (reward_1 - reward_2).item() 
        w2 = weights[1] - step_size * (reward_1 - reward_2).item() 

       
        w1 = max(0.0, w1)
        w2 = max(0.0, w2)
        weight_sum = w1 + w2
        w1 /= weight_sum
        w2 /= weight_sum

        info_nce_loss = self.info_nce_loss(outputs_1, outputs_2)

       
        # occ_loss = ((self.occ_loss(outputs_1, center) + self.occ_loss(outputs_2, center))/2)
        occ_loss = self.w1*(self.occ_loss(outputs_1, center)) +self.w2*(self.occ_loss(outputs_2, center))

        occ_loss.to(self.device)

       
        alpha = 0.5 
        total_loss = alpha * info_nce_loss + (1 - alpha) * occ_loss


        return total_loss

