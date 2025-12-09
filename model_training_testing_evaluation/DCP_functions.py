# import pytorch libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import os
import csv
import copy
import math
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import mse
from scipy import stats
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score

# train the model on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DCP_MLP(torch.nn.Module):
    def __init__(self, input_emb_size, MLP_hl1_size, MLP_hl2_size, N_node, dropout):
        super().__init__()
        # Embedding layer
        self.emb = torch.nn.Embedding(N_node, input_emb_size)
        # MLP layers
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3 * input_emb_size, MLP_hl1_size),  # Adjusted for embedding size
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl1_size, MLP_hl2_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl2_size, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, drug_list1, drug_list2, cond_list):
        predictions = []
        # Directly treat the inputs as representations
        drug_list1_rep = self.emb[drug_list1]
        drug_list2_rep = self.emb[drug_list2]
        cond_list_rep = self.emb[cond_list]
        # Prediction 1
        concat_emb1 = torch.cat((drug_list1_rep, drug_list2_rep, cond_list_rep), dim=1)
        predictions.append(self.linear_relu_stack(concat_emb1))
        # Prediction 2
        concat_emb2 = torch.cat((drug_list2_rep, drug_list1_rep, cond_list_rep), dim=1)
        predictions.append(self.linear_relu_stack(concat_emb2))
        # Compute the final prediction as the mean
        concat_pred = torch.mean(torch.stack(predictions), dim=0)
        concat_pred = torch.squeeze(concat_pred)
        
        return concat_pred

    
class DCP_MLP_avg_emb(torch.nn.Module):
    def __init__(self, input_emb_size, MLP_hl1_size, MLP_hl2_size, N_node, dropout):
        super().__init__()
        self.emb = torch.nn.Embedding(N_node, input_emb_size)
        # MLP
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(2 * input_emb_size, MLP_hl1_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl1_size, MLP_hl2_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl2_size, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, drug_list1, drug_list2, cond_list):
        # Directly treat the inputs as representations
        drug_list1_rep = self.emb[drug_list1]
        drug_list2_rep = self.emb[drug_list2]
        cond_list_rep = self.emb[cond_list]
        # Take the mean of drug embeddings
        avg_drug_rep = (drug_list1_rep + drug_list2_rep) / 2
        # Concatenate with condition embeddings
        concat_rep = torch.cat((avg_drug_rep, cond_list_rep), dim=1)
        # Pass through MLP
        prediction = self.linear_relu_stack(concat_rep)
        prediction = prediction.view(-1)
        
        return prediction
    
    
class DCP_MLP_avg_effect(torch.nn.Module):
    def __init__(self, input_emb_size, MLP_hl1_size, MLP_hl2_size, N_node, dropout):
        super().__init__()
        self.emb = torch.nn.Embedding(N_node, input_emb_size)
        # learning effect
        self.learn_effect_layer = torch.nn.Sequential(
            torch.nn.Linear(3 * input_emb_size, MLP_hl1_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl1_size, MLP_hl2_size)
        )
        # MLP
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(MLP_hl2_size, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, drug_list1, drug_list2, cond_list):
        effects = []
        # Directly treat the inputs as representations
        drug_list1_rep = self.emb[drug_list1]
        drug_list2_rep = self.emb[drug_list2]
        cond_list_rep = self.emb[cond_list]
        # effect 1
        concat_emb1 = torch.cat((drug_list1_rep, drug_list2_rep, cond_list_rep), dim=1)
        effect1 = self.learn_effect_layer(concat_emb1)
        effects.append(effect1)
        # effect 2
        concat_emb2 = torch.cat((drug_list2_rep, drug_list1_rep, cond_list_rep), dim=1)
        effect2 = self.learn_effect_layer(concat_emb2)
        effects.append(effect2)
        # add up the effects
        add_up_effect = torch.sum(torch.stack(effects), dim=0)
        # go into MLP
        prediction = self.linear_relu_stack(add_up_effect)
        prediction = prediction.view(-1)
        
        return prediction


# define the function to build batch
def build_batch(batch_size, triplets):
    if batch_size>=len(triplets):
        return [triplets]
    else:
        shuffled_triplets = random.sample(range(len(triplets)), len(triplets))
        output_list = []
        # the way below will exclude the tail (if not enough for a batch) of shuffled samples but its ok!
        for i in range(0, len(shuffled_triplets)-batch_size+1, batch_size):
            random_order = shuffled_triplets[i:(i + batch_size)]
            random_triplets_batch = [triplets[t] for t in random_order]
            output_list.append(random_triplets_batch)
        return output_list
    
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are float
        targets = targets.float()
        bce_loss = torch.nn.BCELoss(reduction='none')(inputs, targets)
        # Compute the probability that the model predicts the correct class
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        # Apply the modulating factor
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        # Apply the reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
