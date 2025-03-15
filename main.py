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

        self.patience = patience 
        self.verbose = verbose  
        self.counter = 0  
        self.best_score = None  
        self.early_stop = False 
        self.val_loss_min = np.Inf 
        self.delta = delta  
        self.path = path  

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:    
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta: 
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: 
                self.early_stop = True
        else:   
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:  
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 

def train(model, dataloder, device, epoch, tokenizer, optimizer,batch_size):
    num_epochs = epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloder:
           
            graph_data, texts, labels = batch

          
            graph_data = graph_data.to(device)
            print(graph_data.shape)

           
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

          
            optimizer.zero_grad()

            
            loss = model(graph_data, input_ids, attention_mask)
            loss = loss.mean() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            optimizer.step()

           
            running_loss += loss.item()
            running_loss = running_loss / batch_size

       
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloder)}")


def eval(model, dataloder, device, tokenizer):
    model.eval()
    target_all = []
    score_all = []
    with torch.no_grad():
        for batch in dataloder:
           
            graph_data, texts, labels = batch

          
            graph_data = graph_data.to(device)

            labels = labels.to(device)

           
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            loss = model(graph_data, input_ids, attention_mask)

            
            score = loss.flatten()
          

            score_all.append(score.cpu().numpy())
            target_all.append(labels.cpu().numpy())

   
    score_all = np.concatenate(score_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)
    # print(target_all)

    auc = roc_auc_score(target_all, score_all)
   
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(target_all, score_all)

   
    optimal_idx = np.argmax(2 * precision_vals * recall_vals / (precision_vals + recall_vals))
    optimal_threshold = pr_thresholds[optimal_idx]

   
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
   
    parser = argparse.ArgumentParser()

  
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
    
    args = parser.parse_args()
    

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



   
    texts = df['title'].tolist()
    labels = df['label'].tolist()

   
    dataset = CombinedDataset(root='/root/autodl-tmp/tmp/Code/datasets/FakeNewsData', name='gossipcop', feature='spacy',
                              texts=texts, labels=labels, empty=False)
    # dataset = CombinedDataset(root='/root/autodl-tmp/tmp/Code/datasets/FakeNewsData', name='politifact', feature='spacy',
    #                           texts=texts, labels=labels, empty=False)

    first_item = dataset[0]

   
    graph_data, text, label = first_item
    print("Graph Data:", graph_data)
    print("Text:", text)
    print("Label:", label)

   
    train_idx = dataset.train_idx
    test_idx = dataset.test_idx

   
    train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
    test_dataset = torch.utils.data.Subset(dataset, test_idx.tolist())

    
    batch_size = 128

   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


  

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
   
    early_stopping = EarlyStopping(patience=7, verbose=True, path='/root/autodl-tmp/tmp/Code/checkpoint.pt')
 
    for current_epoch in range(epoch):
        print(f"####### Run epoch:{current_epoch + 1} ")

        train(model, train_loader, device, current_epoch + 1, tokenizer, optimizer, batch_size)

       
        auc, precision, recall, f1 = eval(model, test_loader, device, tokenizer)

       
        print(f"Epoch {current_epoch + 1}/{epoch}, "
              f" AUC: {auc}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

       
        early_stopping(-auc, model) 

        
        if early_stopping.early_stop:
            print("Early stopping")
            break

       
        model.load_state_dict(torch.load(early_stopping.path))

        auc_list.append(auc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)


  
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






