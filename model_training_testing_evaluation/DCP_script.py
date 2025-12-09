# import pytorch libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import StepLR
# import other libraries
import re
import os
import csv
import copy
import math
import time
import random
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import mse
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
# import functions from my script
from DCP_functions import DCP_MLP
from DCP_functions import DCP_MLP_avg_emb
from DCP_functions import DCP_MLP_avg_effect
from DCP_functions import build_batch
from DCP_functions import sigmoid
from DCP_functions import FocalLoss


# model parameters
def arg_parse():
    parser = argparse.ArgumentParser(description = 'Model arguments')
    parser.add_argument("--model_name", type=str, help = "model name")
    parser.add_argument("--lr", type=float, help = "learning rate for model training")
    parser.add_argument("--epoch", type=int, help = "number of epoch")
    parser.add_argument("--MLP_hl1_size", type=int, help = "the size of 1st hidden layer in MLP")
    parser.add_argument("--MLP_hl2_size", type=int, help = "the size of 2nd hidden layer in MLP")
    parser.add_argument("--dropout", type=float, help = "dropout rate")
    parser.add_argument("--batch_size", type=int, help = "batch size for model training")
    parser.add_argument("--multi", type=int, help = "the ratio of negative samples : positive samples")
    parser.add_argument("--seed", type=int, help = "the seed for randomness")
    parser.add_argument("--train_partition", type=float, help = "the percentage of triplets in training")
    parser.add_argument("--test_partition", type=float, help = "the percentage of triplets in testing")
    parser.add_argument("--sampling_ratio", type=float, help = "the percentage of negative samplings from the first kind")

    parser.set_defaults(model_name = 'MLP_avg_effect',
                        lr = 0.01,
                        epoch = 200,
                        MLP_hl1_size = 128,
                        MLP_hl2_size = 32,
                        dropout = 0.5,
                        batch_size = 256,
                        multi = 5,
                        seed = 2024,
                        train_partition = 0.8, 
                        test_partition = 0.1, 
                        sampling_ratio = 0.8)
    
    return parser.parse_args()


def train(args):
    
    # settings
    model_name = args.model_name
    lr = args.lr
    epoch = args.epoch
    MLP_hl1_size = args.MLP_hl1_size
    MLP_hl2_size = args.MLP_hl2_size
    dropout = args.dropout
    batch_size = args.batch_size
    multi = args.multi
    seed = args.seed
    train_partition = args.train_partition
    test_partition = args.test_partition
    sampling_ratio = args.sampling_ratio
    # use gpu if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ### directory
    # this is the directory where you save your preprocessing data from clinical trials
    ct_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/clinical_trail_preprocessing/'
    # this is the directory where you have the cleaned drkg embeddings for drugs and conditions
    drkg_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/drkg_embedding_preprocessing/'
    # this is the directory where you save the drugs and conditions embeddings after training and finetuning
    emb_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/embedding_training_and_finetuning/'
    # this is the directory where you save the cdcdb triplets tables
    cdcdb_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/cdcdb_triplets_preprocessing/'
    # this is the directory where you plan to save the model running and evaluation results
    save_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/model_training_testing_evaluation/results/'

    
    #####-----read data from clinical trials-----#####
    # table of clean clinical trials with info of 1 drug and 1 condition
    drug_cond_df = pd.read_csv(ct_dir + 'drug_condition.txt', delimiter='\t')
    # clinical trial triplets
    two_drugs_comb_df = pd.read_csv(ct_dir + 'dcombinations_w_conditions.txt', delimiter='\t')
    # Sort the first two drugs in each row
    two_drugs_comb_df[['drug1', 'drug2']] = two_drugs_comb_df[['drug1', 'drug2']].apply(lambda row: pd.Series(sorted(row)), axis=1)

    
    #####-----Load the embeddings-----#####
    # find the drug and cond that have corresponding embeddings in DRKG dataset
    embedding_drug=pd.read_csv(drkg_dir + 'drkg_filtered_drug_embedding.csv')
    embedding_drug=embedding_drug.set_index('id')
    embedding_drug1=embedding_drug.reindex(drug_cond_df['drug'].unique())
    embedding_drug1=embedding_drug1.dropna()
    embedding_cond=pd.read_csv(drkg_dir + 'drkg_filtered_condition_embedding.csv')
    embedding_cond=embedding_cond.set_index('id')
    embedding_cond1=embedding_cond.reindex(drug_cond_df['condition'].unique())
    embedding_cond1=embedding_cond1.dropna()

    # count the number of nodes of each kind
    n_emb = len(embedding_cond1) + len(embedding_drug1)
    n_drug = len(set(drug_cond_df['drug']))
    n_drug_w_emb = embedding_drug1.shape[0]
    n_cond = len(set(drug_cond_df['condition']))
    n_cond_w_emb = embedding_cond1.shape[0]
    N_node = n_cond + n_drug

    input_emb_size = 400
    # merge the drug embeddings and condition embeddings
    merged_emb = pd.concat([embedding_drug1, embedding_cond1], ignore_index=False)
    # load pretrained drkg embeddings
    pretrained_embedding = torch.nn.Embedding(n_emb, input_emb_size)
    drkg_emb = torch.tensor(merged_emb.values, dtype=torch.float32)
    pretrained_embedding.weight = torch.nn.Parameter(drkg_emb)
    # load the trained embeddings
    embedding_save_path = emb_dir+'trained_emb.npy'
    trained_emb_weights = np.load(embedding_save_path)
    trained_embedding = torch.nn.Embedding(N_node-n_emb, input_emb_size)
    trained_embedding.weight.data.copy_(torch.from_numpy(trained_emb_weights))
    # concat two embeddings layers together
    embeddings = torch.cat((pretrained_embedding.weight, trained_embedding.weight), dim=0) 

    # reindex the existing DRKG embeddings
    emb_id_mapping = {emb_id: i for i, emb_id in enumerate(merged_emb.index)}
    # find all the drugs and conditions without DRKG embeddings
    for i in range(drug_cond_df.shape[0]):
        drug_name = drug_cond_df['drug'][i]
        if drug_name not in emb_id_mapping.keys():
            emb_id_mapping[drug_name] = max(emb_id_mapping.values())+1
    for i in range(drug_cond_df.shape[0]):
        cond_name = drug_cond_df['condition'][i]
        if cond_name not in emb_id_mapping.keys():
            emb_id_mapping[cond_name] = max(emb_id_mapping.values())+1

    # map the drug or cond ids into indices for drug_cond_df
    import warnings
    warnings.filterwarnings("ignore")
    drug_cond_df['drug_idx'] = drug_cond_df['drug'].map(emb_id_mapping)
    drug_cond_df['cond_idx'] = drug_cond_df['condition'].map(emb_id_mapping)

    # find all unique drugs and conditions in clinical trials
    all_drug_list = drug_cond_df['drug'].tolist()
    all_cond_list = drug_cond_df['condition'].tolist()
    unique_drug_list = sorted(list(set(drug_cond_df['drug'].tolist())))
    unique_cond_list = sorted(list(set(drug_cond_df['condition'].tolist())))

    # filter out the combinations we can use from clinical trials
    pick_row = []
    for i in range(two_drugs_comb_df.shape[0]):
        if two_drugs_comb_df['drug1'][i] in all_drug_list:
            if two_drugs_comb_df['drug2'][i] in all_drug_list:
                if two_drugs_comb_df['condition'][i] in all_cond_list:
                    pick_row.append(i)
    pick_comb_df = two_drugs_comb_df.iloc[pick_row,:].reset_index(drop=True)
    pick_comb_df['drug1_idx'] = pick_comb_df['drug1'].map(emb_id_mapping)
    pick_comb_df['drug2_idx'] = pick_comb_df['drug2'].map(emb_id_mapping)
    pick_comb_df['cond_idx'] = pick_comb_df['condition'].map(emb_id_mapping)
    # this is the final clean table for drug combination + condition from clinical trial
    pick_comb_df = pick_comb_df.drop_duplicates().reset_index(drop=True)
    
    #####-----Load CDCDB triplets-----#####
    # read cdcdb dcomb
    cdcdb_dcomb_file = cdcdb_dir+'drugcombo_max2_drugs_conditions_MeSH_IDs.txt'
    cdcdb_dcomb = pd.read_csv(cdcdb_dcomb_file, sep='\t')
    # build the drug comb + condition files
    drug1_list, drug2_list, cond_list = [], [], []
    for i in range(cdcdb_dcomb.shape[0]):
        db_string = cdcdb_dcomb['drugbank_identifiers'][i]
        split_strings = db_string.split(';')
        drug_pair = []
        for s in split_strings:
            drug_pair.extend(re.findall(r'DB\S+', s))
        drug1_list.append(drug_pair[0])
        drug2_list.append(drug_pair[1])
        cond_list.append(cdcdb_dcomb['MESH_CODE'][i])
    # make a new dataframe with 3 columns: drug1, drug2, and condition
    cdcdb_dcomb_cond = pd.DataFrame({
        'drug1': drug1_list,
        'drug2': drug2_list,
        'condition': cond_list
    })
    
    ### only include the triplets that have all info in clinical trials 1 drug -> 1 condition data
    all_drug_list = drug_cond_df['drug'].tolist()
    all_cond_list = drug_cond_df['condition'].tolist()
    # generate the dataframe for rows that three elements are all in the drug-cond-data
    pick_row_nona = [i for i in range(cdcdb_dcomb_cond.shape[0]) if not isinstance(cond_list[i], float) or not math.isnan(cond_list[i])]
    pick_row = [i for i in pick_row_nona if drug1_list[i] in all_drug_list if drug2_list[i] in all_drug_list if cond_list[i] in all_cond_list]
    cdcdb_comb = cdcdb_dcomb_cond.iloc[pick_row].reset_index(drop=True)
    
    ### clean the cdcdb triplets data
    # Sort the first two elements in each row
    cdcdb_comb[['drug1', 'drug2']] = cdcdb_comb[['drug1', 'drug2']].apply(lambda row: pd.Series(sorted(row)), axis=1)
    # remove duplicated rows
    cdcdb_comb = cdcdb_comb.drop_duplicates().reset_index(drop=True)
    # remove the combination if two drugs are identical
    cdcdb_comb = cdcdb_comb[cdcdb_comb['drug1'] != cdcdb_comb['drug2']].reset_index(drop=True)
    # mapped drugs and conditions to index
    cdcdb_comb['drug1_idx'] = cdcdb_comb['drug1'].map(emb_id_mapping)
    cdcdb_comb['drug2_idx'] = cdcdb_comb['drug2'].map(emb_id_mapping)
    cdcdb_comb['cond_idx'] = cdcdb_comb['condition'].map(emb_id_mapping)
    print(f"There are {cdcdb_comb.shape[0]} triplets in cdcdb after final cleaning.")
    # combine triplets from cdcdb and clinical trial
    print(f"There are {pick_comb_df.shape[0]} triplets in drug-application dataset.")
    combined_df = pd.concat([cdcdb_comb, pick_comb_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    print(f"After combining two datasets, there are {combined_df.shape[0]} unique triplets.\n")


    #####-----build training, testing, and validation sets-----#####
    all_drug_in_comb_list = combined_df['drug1_idx'].tolist() + combined_df['drug2_idx'].tolist()
    unique_drug_in_comb_list = sorted(list(set(all_drug_in_comb_list)))
    print(f'There are {len(unique_drug_in_comb_list)} unique drugs in drug combinations!')

    # shuffle the triplets
    seed = seed
    train_partition = train_partition
    test_partition = test_partition
    ### triplets are already sorted here!!!
    all_comb_triplets = [
        sorted([drug1, drug2]) + [cond]
        for drug1, drug2, cond in zip(combined_df['drug1_idx'], combined_df['drug2_idx'], combined_df['cond_idx'])
    ]

    # Split the list into training, testing, and validation sets
    training_triplets, tmp_triplets = train_test_split(all_comb_triplets, test_size=(1-train_partition), random_state=seed)
    testing_triplets, validation_triplets = train_test_split(tmp_triplets, test_size=(1-train_partition-test_partition)/(1-train_partition), random_state=seed)

    unique_drug_in_training = sorted(list(set(list(np.array(training_triplets).T[0]) + list(np.array(training_triplets).T[1]))))
    unique_drug_in_testing = sorted(list(set(list(np.array(testing_triplets).T[0]) + list(np.array(testing_triplets).T[1]))))
    unique_drug_in_validation = sorted(list(set(list(np.array(validation_triplets).T[0]) + list(np.array(validation_triplets).T[1]))))                               
    print(f'There are {len(training_triplets)} triplets in training with {len(unique_drug_in_training)} unique drugs.')
    print(f'There are {len(testing_triplets)} triplets in testing with {len(unique_drug_in_testing)} unique drugs.')
    print(f'There are {len(validation_triplets)} triplets in validation with {len(unique_drug_in_validation)} unique drugs.\n')

    
    ### other info
    # build a dictionary with keys are conditions and values are drugs connected to the corresponding condition
    cond_dict = {}
    cond_list = sorted(list(set(drug_cond_df['cond_idx'])))
    for cond in cond_list:
        cond_dict[cond] = sorted(drug_cond_df[drug_cond_df['cond_idx']==cond]['drug_idx'].tolist())

    # find all the drug idx and condition idx
    all_unique_drug_idx_list = sorted(set(drug_cond_df['drug_idx'].tolist()))
    # all_unique_drug_idx_list_in_training = sorted(set(training_drug_cond_df['drug_idx'].tolist()))
    all_unique_cond_idx_list = sorted(set(drug_cond_df['cond_idx'].tolist()))    
    
    
    #####-----functions-----#####

    # negative samples of first kind (fix one drug and condition then replace second drug with an alternative drug)
    def sampling_part1(target_triplets, pos_triplets, condition_dict, all_unique_drug_idx_list, multi, sampling_ratio=0.8):
        gen_triplets = []
        covered_cond_idx = list(set(np.array(target_triplets).T[2]))
        for cond_idx in covered_cond_idx:
            # find the subset of the existing triplets related to this condition
            sub_pos_triplets = [t for t in pos_triplets if t[2]==cond_idx]
            # existing triplets are already sorted
            existed_comb = sub_pos_triplets.copy()
            # find the subset of the target triplets related to this condition
            sub_target_triplets = [t for t in target_triplets if t[2]==cond_idx]
            total_n = int(len(sub_target_triplets)*multi*sampling_ratio)
            # pick a random drug from each of the triplets
            drugs_in_target_triplets = list(np.array(sub_target_triplets).T[0])+list(np.array(sub_target_triplets).T[1])
            pick_drugs = random.choices(drugs_in_target_triplets, k=total_n)
            # pick a drug from the drugs connected to the same cond_idx or a totally random drug
            connnected_drug_node = condition_dict[cond_idx].copy()
            rest_drug_node = [x for x in all_unique_drug_idx_list if x not in connnected_drug_node]
            if len(connnected_drug_node)>=total_n:
                random_drugs = random.sample(connnected_drug_node, total_n)
            else:
                random_drugs = connnected_drug_node + random.choices(rest_drug_node, k=(total_n-len(connnected_drug_node)))
                random.shuffle(random_drugs)
            # cleaning process
            tmp_triplet1 = [sorted([pick_drugs[i], random_drugs[i]]) + [cond_idx] for i in range(total_n)]
            tmp_triplet2 = [list(tpl) for tpl in set(tuple(lst) for lst in tmp_triplet1)]
            # filter out existing triplets
            tmp_triplet3 = [x for x in tmp_triplet2 if x not in existed_comb] 
            # make sure first drug is not the same as second drug
            sub_gen_triplets = [x for x in tmp_triplet3 if x[0]!=x[1]]
            # contiune to find more
            remain_n = total_n - len(sub_gen_triplets)
            while remain_n>0:
                pick_drugs = random.choices(drugs_in_target_triplets, k=remain_n)
                random_drugs = random.choices(rest_drug_node, k=remain_n)
                # filters
                tmp_triplet1 = [sorted([pick_drugs[i], random_drugs[i]]) + [cond_idx] for i in range(remain_n)]
                tmp_triplet2 = [list(tpl) for tpl in set(tuple(lst) for lst in tmp_triplet1)]
                tmp_triplet3 = [x for x in tmp_triplet2 if x not in existed_comb+sub_gen_triplets] 
                tmp_triplet4 = [x for x in tmp_triplet3 if x[0]!=x[1]]
                sub_gen_triplets = sub_gen_triplets + tmp_triplet4
                remain_n = total_n - len(sub_gen_triplets)
            # append the sub_gen_triplets to gen_triplets
            gen_triplets = gen_triplets + sub_gen_triplets

        return gen_triplets

    # negative samples of second kind (fix condition then replace drug combination with an alternative drug combination)
    def sampling_part2(target_triplets, pos_triplets, condition_dict, all_unique_cond_idx_list, multi, sampling_ratio=0.2):
        gen_triplets = []
        for triplet in target_triplets:
            drug1 = triplet[0]
            drug2 = triplet[1]
            cond = triplet[2]
            total_n = int(multi*sampling_ratio)
            # find existing triplets
            existing_triplets = [x for x in pos_triplets if (x[0]==drug1 and x[1]==drug2) or (x[1]==drug2 and x[1]==drug1)]
            applied_cond_list = [x[2] for x in existing_triplets]
            # find the rest of the conditions
            rest_cond_node = [x for x in all_unique_cond_idx_list if x not in applied_cond_list]
            if len(rest_cond_node)>=total_n:
                random_cond_list = random.sample(rest_cond_node, total_n)
            else:
                random_cond_list = random.choices(rest_cond_node, k=total_n)
            tmp_triplets = [[drug1, drug2] + [random_cond] for random_cond in random_cond_list]
            gen_triplets = gen_triplets + tmp_triplets

        return gen_triplets

    # merged negative sampling function
    def sampling_method(target_triplets, pos_triplets, condition_dict, all_unique_drug_idx_list, all_unique_cond_idx_list, multi, sampling_ratio=0.8):
        if sampling_ratio==1:
            gen_triplets = sampling_part1(target_triplets, pos_triplets, condition_dict, all_unique_drug_idx_list, multi, sampling_ratio)
        elif sampling_ratio==0:
            gen_triplets = sampling_part2(target_triplets, pos_triplets, condition_dict, all_unique_cond_idx_list, multi, 1-sampling_ratio)
        else:
            gen_triplets1 = sampling_part1(target_triplets, pos_triplets, condition_dict, all_unique_drug_idx_list, multi, sampling_ratio)
            gen_triplets2 = sampling_part2(target_triplets, pos_triplets, condition_dict, all_unique_cond_idx_list, multi, 1-sampling_ratio)
            gen_triplets = gen_triplets1 + gen_triplets2

        return gen_triplets

    # define function that used in model training and evaluation
    def train(model, model_name, triplets, real):
        model.train()
        optimizer.zero_grad()
        # freeze the embedding layer during training
        model.emb.requires_grad = False
        # prediction
        drug_list1, drug_list2, cond_list = np.array(triplets).T
        pred = model.forward(drug_list1, drug_list2, cond_list)
        loss = criterion(pred, real)
        loss.backward()
        # clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
        optimizer.step()

        return loss

    def test(multi):
        # test the model on training triplets
        model.eval()
        negative_triplets = sampling_method(training_triplets, training_triplets, cond_dict, all_unique_drug_idx_list, all_unique_cond_idx_list, multi, sampling_ratio)
        # print(f"negative samples for training triplets in testing: {len(negative_triplets)}")
        all_pairs = training_triplets + negative_triplets
        drug_list1, drug_list2, cond_list = np.array(all_pairs).T
        pred = model.forward(drug_list1, drug_list2, cond_list)
        real = torch.tensor([1] * len(training_triplets) + [0] * len(negative_triplets)).float()
        # compute the loss and accuracy
        train_loss = criterion(pred, real).detach().numpy()
        prediction = pred.detach().numpy()
        # compute the auc score
        train_roc_auc = roc_auc_score(real, prediction)
        train_pr_auc = average_precision_score(real, prediction)

        # testing the model on testing triplets
        all_pairs = testing_triplets + testing_negative_triplets
        drug_list1, drug_list2, cond_list = np.array(all_pairs).T
        pred = model.forward(drug_list1, drug_list2, cond_list)
        real = torch.tensor([1] * len(testing_triplets) + [0] * len(testing_negative_triplets)).float()
        # compute the loss and accuracy
        test_loss = criterion(pred, real).detach().numpy()
        prediction = pred.detach().numpy()
        # compute the auc score
        test_roc_auc = roc_auc_score(real, prediction)
        test_pr_auc = average_precision_score(real, prediction) 

        # validating the model on validation set
        all_pairs = validation_triplets + validation_negative_triplets
        drug_list1, drug_list2, cond_list = np.array(all_pairs).T
        pred = model.forward(drug_list1, drug_list2, cond_list)
        real = torch.tensor([1] * len(validation_triplets) + [0] * len(validation_negative_triplets)).float()
        # compute the loss and accuracy
        valid_loss = criterion(pred, real).detach().numpy()
        prediction = pred.detach().numpy()
        # compute the auc score
        valid_roc_auc = roc_auc_score(real, prediction)
        valid_pr_auc = average_precision_score(real, prediction)     

        return train_loss, train_roc_auc, train_pr_auc, test_loss, test_roc_auc, test_pr_auc, valid_loss, valid_roc_auc, valid_pr_auc

    
    #####-----Running DCP model-----#####
    switch = True
    if switch:
        # record the time
        start_time = time.time()
        input_emb_size = 400
        N_node = n_cond + n_drug
        # gradient clipping and value clipping
        min_clip_value = embeddings.min().min()
        max_clip_value = embeddings.max().max()
        gradient_clip_value = 1.0

        if model_name=='MLP_avg_effect':
            model = DCP_MLP_avg_effect(input_emb_size, MLP_hl1_size, MLP_hl2_size, N_node, dropout)

        # use embeddings after finetuning
        embedding_save_path = emb_dir+'finetuned_emb_1.npy'
        trained_emb_weights = np.load(embedding_save_path)
        pretrained_embedding = torch.nn.Embedding(N_node, input_emb_size)
        pretrained_embedding.weight.data.copy_(torch.from_numpy(trained_emb_weights))
        embeddings = pretrained_embedding.weight
        node_embed = torch.tensor(embeddings, dtype=torch.float32)
        # freeze embeddings during training and finetuning
        model.emb = torch.nn.Parameter(node_embed)
        model.emb.requires_grad = False

        # criterion and optimizer
        criterion = torch.nn.BCELoss(reduction='mean')
        # criterion = FocalLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Reduces LR by a factor of 0.1 every 10 epochs
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

        # lists of error
        train_loss_list, train_rocauc_list, train_prauc_list = [], [], []
        test_loss_list, test_rocauc_list, test_prauc_list = [], [], []
        valid_loss_list, valid_rocauc_list, valid_prauc_list = [], [], []

        # positive pairs and testing positive pairs
        training_triplets = training_triplets
        testing_triplets = testing_triplets
        validation_triplets = validation_triplets

        # pre-define a set of negative triplets
        testing_negative_triplets = sampling_method(testing_triplets, training_triplets+testing_triplets, cond_dict, all_unique_drug_idx_list, all_unique_cond_idx_list, multi, sampling_ratio)
        validation_negative_triplets = sampling_method(validation_triplets, training_triplets+testing_triplets+validation_triplets, cond_dict, all_unique_drug_idx_list, all_unique_cond_idx_list, multi, sampling_ratio)

        # Initial error
        train_loss, train_roc_auc, train_pr_auc, test_loss, test_roc_auc, test_pr_auc, valid_loss, valid_roc_auc, valid_pr_auc = test(multi)

        # append the result
        train_loss_list.append(train_loss)
        train_rocauc_list.append(train_roc_auc)
        train_prauc_list.append(train_pr_auc)
        test_loss_list.append(test_loss)
        test_rocauc_list.append(test_roc_auc)
        test_prauc_list.append(test_pr_auc)
        valid_loss_list.append(valid_loss)
        valid_rocauc_list.append(valid_roc_auc)
        valid_prauc_list.append(valid_pr_auc)

        print(f'Initial error:\n' + 
              f'Train Loss:{train_loss:.4f}, roc_auc:{train_roc_auc:.4f}, pr_auc:{train_pr_auc:.4f},\n' +
              f'Test Loss:{test_loss:.4f}, roc_auc:{test_roc_auc:.4f}, pr_auc:{test_pr_auc:.4f},\n' +
              f'Validation Loss:{valid_loss:.4f}, roc_auc:{valid_roc_auc:.4f}, pr_auc:{valid_pr_auc:.4f}.\n')

        # print the running time
        now_time = time.time()
        total_time = now_time - start_time
        print('Running time: {:.2f} seconds'.format(total_time), "\n")

        # training iteration     
        for e in range(epoch):
            # build batchs of training
            training_triplets_batch_list = build_batch(batch_size, training_triplets)
            # model training
            for triplets_batch in training_triplets_batch_list:
                negative_sampling_triplets = sampling_method(triplets_batch, training_triplets, cond_dict, all_unique_drug_idx_list, all_unique_cond_idx_list, multi, sampling_ratio)
                real = torch.tensor([1] * len(triplets_batch) + [0] * len(negative_sampling_triplets)).float()
                triplets = triplets_batch + negative_sampling_triplets
                batch_loss = train(model, model_name, triplets, real)

            if (e+1) % 1 == 0:
                train_loss, train_roc_auc, train_pr_auc, test_loss, test_roc_auc, test_pr_auc, valid_loss, valid_roc_auc, valid_pr_auc = test(multi)
                # append the result
                train_loss_list.append(train_loss)
                train_rocauc_list.append(train_roc_auc)
                train_prauc_list.append(train_pr_auc)
                test_loss_list.append(test_loss)
                test_rocauc_list.append(test_roc_auc)
                test_prauc_list.append(test_pr_auc)
                valid_loss_list.append(valid_loss)
                valid_rocauc_list.append(valid_roc_auc)
                valid_prauc_list.append(valid_pr_auc)

                print(f'Epoch:{(e+1):03d},\n' + 
                      f'Train Loss:{train_loss:.4f}, roc_auc:{train_roc_auc:.4f}, pr_auc:{train_pr_auc:.4f},\n' +
                      f'Test Loss:{test_loss:.4f}, roc_auc:{test_roc_auc:.4f}, pr_auc:{test_pr_auc:.4f},\n' + 
                      f'Validation Loss:{valid_loss:.4f}, roc_auc:{valid_roc_auc:.4f}, pr_auc:{valid_pr_auc:.4f}.\n')
                
                # print the running time
                now_time = time.time()
                total_time = now_time - start_time
                print('Running time: {:.2f} seconds'.format(total_time), "\n")
                
            scheduler.step()
            print(f"Epoch {e + 1}: Learning rate is {scheduler.get_last_lr()}")

        # save the model
        model_name = model_name
        save_dir = save_dir
        # Save the model arthitecture and weights
        m_name = f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_architecture_after_finetuning.pth'
        w_name = f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_weights_after_finetuning.pth'
        # torch.save(model, save_dir+m_name)
        # torch.save(model.state_dict(), save_dir+w_name)


        #####----save the training and testing curves----#####
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        # fig.suptitle(f"Loss and Accuracy curve for Training;\n" + 
        #              f"model name: {model_name}, lr: {lr}, negative samples:positive samples={multi}:1", fontsize=12)

        axes[0].plot(range(0, 1+epoch, 1), train_loss_list, color='black', label="BCELoss")
        axes[0].plot(range(0, 1+epoch, 1), train_rocauc_list, color='orange', label="ROC AUC")
        axes[0].plot(range(0, 1+epoch, 1), train_prauc_list, color='blue', label="PR AUC")
        axes[0].set_title('Training set', fontsize=15)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, linestyle='--')

        axes[1].plot(range(0, 1+epoch, 1), test_loss_list, color='black', label="BCELoss")
        axes[1].plot(range(0, 1+epoch, 1), test_rocauc_list, color='orange', label="ROC AUC")
        axes[1].plot(range(0, 1+epoch, 1), test_prauc_list, color='blue', label="PR AUC")
        axes[1].set_title('Testing set', fontsize=15)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, linestyle='--')

        axes[2].plot(range(0, 1+epoch, 1), valid_loss_list, color='black', label="BCELoss")
        axes[2].plot(range(0, 1+epoch, 1), valid_rocauc_list, color='orange', label="ROC AUC")
        axes[2].plot(range(0, 1+epoch, 1), valid_prauc_list, color='blue', label="PR AUC")
        axes[2].set_title('Validation set', fontsize=15)
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, linestyle='--')

        axes[2].legend(loc='upper right', fontsize='small')

        # Show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_dir = save_dir
        plt.savefig(save_dir+f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_loss_accuracy_curve_after_finetuning.png')

    
    #####-----trained model loading-----#####
    # read the saved model
    model_name = model_name
    save_dir = save_dir
    m_name = f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_architecture_after_finetuning.pth'
    w_name = f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_weights_after_finetuning.pth'
    test_model = torch.load(save_dir + m_name)
    test_model.load_state_dict(torch.load(save_dir + w_name))
    test_model.eval()

    
    #####-----second drug ranking evaluation-----#####
    switch = True
    if switch:
        comb_ranking = combined_df.copy()
        comb_ranking['type'] = ''
        comb_ranking['drug1_rank_in_[drug2_condition]'] = -1
        comb_ranking['drug2_rank_in_[drug1_condition]'] = -1
        # assign the type: train, test, or validation
        for i in range(comb_ranking.shape[0]):
            drug1_idx = comb_ranking['drug1_idx'][i]
            drug2_idx = comb_ranking['drug2_idx'][i]
            cond_idx = comb_ranking['cond_idx'][i]
            if drug1_idx < drug2_idx:
                triplet = [drug1_idx, drug2_idx, cond_idx]
            else:
                triplet = [drug2_idx, drug1_idx, cond_idx]
            if triplet in training_triplets:
                comb_ranking['type'][i] = 'train'
            elif triplet in testing_triplets:
                comb_ranking['type'][i] = 'test'
            else:
                comb_ranking['type'][i] = 'validation'

        # Precompute Combined Triplets
        all_triplets = training_triplets + testing_triplets + validation_triplets
        all_triplets = np.array(all_triplets)

        # Precompute unique drug sets
        all_unique_drug_set = set(all_unique_drug_idx_list)

        # Predefine the output arrays with integer data type
        drug2_rank_results = np.zeros(comb_ranking.shape[0], dtype=int)
        drug1_rank_results = np.zeros(comb_ranking.shape[0], dtype=int)

        batch = 500  # Number of iterations per epoch
        start_time = time.time()

        print("-----Start ranking computation-----")
        for batch_start in range(0, comb_ranking.shape[0], batch):
            # Initialize batch data
            batch_triplets_drug2 = []
            batch_triplets_drug1 = []
            batch_num_triplets_drug2 = []
            batch_num_triplets_drug1 = []
            batch_indices = range(batch_start, min(batch_start + batch, comb_ranking.shape[0]))

            for i in batch_indices:
                # Extract drug and condition indices for current row
                drug1_idx = comb_ranking['drug1_idx'][i]
                drug2_idx = comb_ranking['drug2_idx'][i]
                cond_idx = comb_ranking['cond_idx'][i]

                # **Compute drug2_rank_in_[drug1_condition]**
                # Filter triplets by condition and anchor drug (drug1)
                condition_mask = (all_triplets[:, 2] == cond_idx)
                drug_mask = (all_triplets[:, 0] == drug1_idx) | (all_triplets[:, 1] == drug1_idx)
                existing_triplets_by_cond_by_drug2 = all_triplets[condition_mask & drug_mask]

                # Extract drug partners for drug1
                drug_list1, drug_list2 = existing_triplets_by_cond_by_drug2[:, 0], existing_triplets_by_cond_by_drug2[:, 1]
                existing_partner_set = set(drug_list1).union(drug_list2) - {drug1_idx}

                # Determine candidate drug list for drug2 ranking
                drug_to_filter_out = {drug2_idx, drug1_idx}.union(existing_partner_set)
                other_candidate_drug_list = list(all_unique_drug_set - drug_to_filter_out)
                all_candidate_drug_list = [drug2_idx] + other_candidate_drug_list

                # Collect triplets for the batch
                possible_triplets_drug2 = [[drug1_idx, drug, cond_idx] for drug in all_candidate_drug_list]
                batch_triplets_drug2.extend(possible_triplets_drug2)
                batch_num_triplets_drug2.append(len(possible_triplets_drug2))

                # **Compute drug1_rank_in_[drug2_condition]**
                # Filter triplets by condition and anchor drug (drug2)
                drug_mask = (all_triplets[:, 0] == drug2_idx) | (all_triplets[:, 1] == drug2_idx)
                existing_triplets_by_cond_by_drug1 = all_triplets[condition_mask & drug_mask]

                # Extract drug partners for drug2
                drug_list1, drug_list2 = existing_triplets_by_cond_by_drug1[:, 0], existing_triplets_by_cond_by_drug1[:, 1]
                existing_partner_set = set(drug_list1).union(drug_list2) - {drug2_idx}

                # Determine candidate drug list for drug1 ranking
                drug_to_filter_out = {drug1_idx, drug2_idx}.union(existing_partner_set)
                other_candidate_drug_list = list(all_unique_drug_set - drug_to_filter_out)
                all_candidate_drug_list = [drug1_idx] + other_candidate_drug_list

                # Collect triplets for the batch
                possible_triplets_drug1 = [[drug2_idx, drug, cond_idx] for drug in all_candidate_drug_list]
                batch_triplets_drug1.extend(possible_triplets_drug1)
                batch_num_triplets_drug1.append(len(possible_triplets_drug1))

            # Convert batch triplets to NumPy arrays for prediction
            batch_triplets_drug2 = np.array(batch_triplets_drug2)
            drug_list1_drug2, drug_list2_drug2, cond_list_drug2 = batch_triplets_drug2.T

            batch_triplets_drug1 = np.array(batch_triplets_drug1)
            drug_list1_drug1, drug_list2_drug1, cond_list_drug1 = batch_triplets_drug1.T

            # Run the model on the entire batch for both rankings
            batch_predictions_drug2 = test_model.forward(drug_list1_drug2, drug_list2_drug2, cond_list_drug2)
            batch_predictions_drug1 = test_model.forward(drug_list1_drug1, drug_list2_drug1, cond_list_drug1)

            # Split predictions and calculate ranks for each iteration
            start_idx_drug2 = 0
            start_idx_drug1 = 0

            for idx, i in enumerate(batch_indices):
                # **Rank for drug2**
                num_triplets_drug2 = batch_num_triplets_drug2[idx]
                predictions_drug2 = batch_predictions_drug2[start_idx_drug2:start_idx_drug2 + num_triplets_drug2]
                start_idx_drug2 += num_triplets_drug2

                # Detach and convert to NumPy
                predictions_drug2 = predictions_drug2.detach().cpu().numpy()
                ranks_drug2 = rankdata(-predictions_drug2, method='ordinal')
                drug2_rank_results[i] = int(ranks_drug2[0])

                # **Rank for drug1**
                num_triplets_drug1 = batch_num_triplets_drug1[idx]
                predictions_drug1 = batch_predictions_drug1[start_idx_drug1:start_idx_drug1 + num_triplets_drug1]
                start_idx_drug1 += num_triplets_drug1

                # Detach and convert to NumPy
                predictions_drug1 = predictions_drug1.detach().cpu().numpy()
                ranks_drug1 = rankdata(-predictions_drug1, method='ordinal')
                drug1_rank_results[i] = int(ranks_drug1[0])

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"time for finishing executing {i+1} triplets: {int(execution_time)} seconds")

        # Store results in the dataframe
        comb_ranking['drug2_rank_in_[drug1_condition]'] = drug2_rank_results
        comb_ranking['drug1_rank_in_[drug2_condition]'] = drug1_rank_results

        print("-----Ranking computation completed-----")

        # visualization
        triplet_types = ['train', 'test', 'validation']
        subplot_names = ['Training set', 'Testing set', 'Validation set']
        proportions_list = []
        rank_list_sizes = []  # To store the size of rank_list for each group
        max_proportion = 0

        # Process data for each triplet type
        for triplet_type in triplet_types:
            rank_list1 = comb_ranking[comb_ranking['type'] == triplet_type]['drug1_rank_in_[drug2_condition]'].tolist()
            rank_list2 = comb_ranking[comb_ranking['type'] == triplet_type]['drug2_rank_in_[drug1_condition]'].tolist()
            rank_list = rank_list1 + rank_list2

            rank_list_sizes.append(len(rank_list))  # Store the size of rank_list

            bins = [1, 4, 11, 101, 501, max(rank_list) + 1]
            freq = [0] * (len(bins) - 1)
            for num in rank_list:
                for i in range(len(bins) - 1):
                    if bins[i] <= num < bins[i + 1]:
                        freq[i] += 1
                        break
            total = sum(freq)
            proportions = [f / total for f in freq]
            proportions_list.append(proportions)
            max_proportion = max(max_proportion, max(proportions))

        # Plot the histograms in a row
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, dpi=300)
        labels = ['1-3', '4-10', '11-100', '101-500', '501+']
        fig.suptitle("Second Drug Ranking Evaluation", fontsize=20)

        for i, ax in enumerate(axes):
            proportions = proportions_list[i]
            ax.bar(labels, proportions, width=0.5)
            ax.set_title(f"{subplot_names[i]}", fontsize=16)
            ax.set_xlabel("Ranks", fontsize=16)
            if i == 0:  # Only label y-axis on the first subplot
                ax.set_ylabel("Proportion", fontsize=16)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_ylim(0, 1)
            for j, p in enumerate(proportions):
                ax.text(j, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=14)

        plt.tight_layout()
        plt.show()

        save_dir = save_dir
        # plt.savefig(save_dir+f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_second_drug_ranking_evaluation.png')
    
    
    #####-----rank of existing triplets among all possible negative triplets by condition-----#####
    switch = True
    if switch:
        def advanced_calculate_rank_percentile_list(A, B):
            B_sorted = np.sort(B)
            c1, c2 = 0, 0
            rank_percentiles = []
            for element in A:
                # Find the actual rank of the element in B
                position = np.searchsorted(B_sorted, element, side='left')
                rank_position = len(B) - position
                # Find the percentile of the element in B
                percentile = position / len(B)
                rank_percentile = 1 - percentile
                # conditions
                if rank_position <= 9:
                    c1 = c1+1
                elif rank_position <=49:
                    c2 = c2+1
                else:
                    rank_percentiles.append(rank_percentile)

            return c1, c2, rank_percentiles

        # find all the drug idx and condition idx
        all_unique_drug_idx_list = sorted(set(drug_cond_df['drug_idx'].tolist()))
        all_unique_cond_idx_list = sorted(set(drug_cond_df['cond_idx'].tolist()))
        existing_triplets = training_triplets+testing_triplets+validation_triplets

        rank_percentile_list_training_triplets = []
        rank_percentile_list_testing_triplets = []
        rank_percentile_list_validation_triplets = []
        top10_rank_training_triplets, top50_rank_training_triplets = 0, 0
        top10_rank_testing_triplets, top50_rank_testing_triplets = 0, 0
        top10_rank_validation_triplets, top50_rank_validation_triplets = 0, 0

        # all the combinations of any two drugs
        drug_combinations = [list(pair) for pair in itertools.combinations(all_unique_drug_idx_list, 2)]

        covered_cond_idx = list(set(np.array(existing_triplets).T[2]))
        print(f"There are {len(covered_cond_idx)} conditions in existing triplets")
        
        for cond_idx in covered_cond_idx:
            sub_training_triplets = [t for t in training_triplets if t[2]==cond_idx]
            sub_testing_triplets = [t for t in testing_triplets if t[2]==cond_idx]
            sub_validation_triplets = [t for t in validation_triplets if t[2]==cond_idx]
            sub_existing_triplets = sub_training_triplets + sub_testing_triplets + sub_validation_triplets
            # generate all negative triplets
            tmp_triplet1 = [pair + [cond_idx] for pair in drug_combinations]
            # filter out existing triplets
            tmp_triplet = [x for x in tmp_triplet1 if x not in sub_existing_triplets]
            
            # Adjust batch size based on memory
            negative_samples_pred_list = []
            bs = int(len(tmp_triplet)/5)+1  
            for i in range(0, len(tmp_triplet), bs):
                batch = tmp_triplet[i:i+bs]
                drug_list1, drug_list2, cond_list = np.array(batch).T
                pred = test_model.forward(drug_list1, drug_list2, cond_list)
                negative_samples_pred_list.extend(pred.detach().cpu().numpy())
            
            # print(len(negative_samples_pred_list))
            # compute the prediction of training triplets
            if len(sub_training_triplets)>0:
                drug_list1, drug_list2, cond_list = np.array(sub_training_triplets).T
                pred = test_model.forward(drug_list1, drug_list2, cond_list)
                pred_list = np.atleast_1d(pred.detach().cpu().numpy())
                c1, c2, rank_percentiles = advanced_calculate_rank_percentile_list(pred_list, negative_samples_pred_list)
                rank_percentile_list_training_triplets = rank_percentile_list_training_triplets + rank_percentiles
                top10_rank_training_triplets = top10_rank_training_triplets + c1
                top50_rank_training_triplets = top50_rank_training_triplets + c2
            # compute the prediction of testing triplets
            if len(sub_testing_triplets)>0:
                drug_list1, drug_list2, cond_list = np.array(sub_testing_triplets).T
                pred = test_model.forward(drug_list1, drug_list2, cond_list)
                pred_list = np.atleast_1d(pred.detach().cpu().numpy())
                c1, c2, rank_percentiles = advanced_calculate_rank_percentile_list(pred_list, negative_samples_pred_list)
                rank_percentile_list_testing_triplets = rank_percentile_list_testing_triplets + rank_percentiles
                top10_rank_testing_triplets = top10_rank_testing_triplets + c1
                top50_rank_testing_triplets = top50_rank_testing_triplets + c2
            # compute the prediction of validation triplets
            if len(sub_validation_triplets)>0:
                drug_list1, drug_list2, cond_list = np.array(sub_validation_triplets).T
                pred = test_model.forward(drug_list1, drug_list2, cond_list)
                pred_list = np.atleast_1d(pred.detach().cpu().numpy())
                c1, c2, rank_percentiles = advanced_calculate_rank_percentile_list(pred_list, negative_samples_pred_list)
                rank_percentile_list_validation_triplets = rank_percentile_list_validation_triplets + rank_percentiles
                top10_rank_validation_triplets = top10_rank_validation_triplets + c1
                top50_rank_validation_triplets = top50_rank_validation_triplets + c2

        # Define bins
        bins = [0, 0.0001, 0.001, 0.01, 0.1, 1]
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True, dpi=300)
        
        # First plot (Training triplets)
        hist, bin_edges = np.histogram(rank_percentile_list_training_triplets, bins=bins)
        hist = np.array([top10_rank_training_triplets] + [top50_rank_training_triplets] + list(hist))
        bars0 = axs[0].bar(range(len(hist)), hist/hist.sum(), width=0.8, edgecolor='black', align='center')
        axs[0].set_xticks(range(len(hist)))
        axs[0].set_xticklabels(['1~10', '11~50', '51~0.01%(188)', '0.01%-0.1%', '0.1%-1%', '1%-10%', '>10%'], fontsize=13, rotation=45)
        axs[0].set_ylabel('Proportion', fontsize=16)
        axs[0].set_title(f'Training set', fontsize=16)
        axs[0].set_ylim(0, 1.0)
        
        # Modify font size for y-axis ticks (the tick labels on the y-axis)
        for label in axs[0].get_yticklabels():
            label.set_fontsize(14)  # Change the fontsize here
        # Add percentage on top of each bar in the first plot
        for bar in bars0:
            height = bar.get_height()
            height_pos = height + 0.004
            axs[0].text(bar.get_x() + bar.get_width() / 2, height_pos, f'{height*100:.1f}%', ha='center', va='bottom', fontsize=13)
        
        # Second plot (Testing triplets)
        hist, bin_edges = np.histogram(rank_percentile_list_testing_triplets, bins=bins)
        hist = np.array([top10_rank_testing_triplets] + [top50_rank_testing_triplets] + list(hist))
        bars1 = axs[1].bar(range(len(hist)), hist/hist.sum(), width=0.8, edgecolor='black', align='center')
        axs[1].set_xticks(range(len(hist)))
        axs[1].set_xticklabels(['1~10', '11~50', '51~0.01%(188)', '0.01%-0.1%', '0.1%-1%', '1%-10%', '>10%'], fontsize=13, rotation=45)
        axs[1].set_title(f'Testing set', fontsize=16)
        axs[1].set_ylim(0, 1.0)
        
        for label in axs[1].get_yticklabels():
            label.set_fontsize(14)  # Change the fontsize here
        # Add percentage on top of each bar in the second plot
        for bar in bars1:
            height = bar.get_height()
            height_pos = bar.get_height() + 0.004
            axs[1].text(bar.get_x() + bar.get_width() / 2, height_pos, f'{height*100:.1f}%', ha='center', va='bottom', fontsize=13)
        
        # Second plot (Validation triplets)
        hist, bin_edges = np.histogram(rank_percentile_list_validation_triplets, bins=bins)
        hist = np.array([top10_rank_validation_triplets] + [top50_rank_validation_triplets] + list(hist))
        bars2 = axs[2].bar(range(len(hist)), hist/hist.sum(), width=0.8, edgecolor='black', align='center')
        axs[2].set_xticks(range(len(hist)))
        axs[2].set_xticklabels(['1~10', '11~50', '51~0.01%(188)', '0.01%-0.1%', '0.1%-1%', '1%-10%', '>10%'], fontsize=13, rotation=45)
        axs[2].set_title(f'Validation set', fontsize=16)
        axs[2].set_ylim(0, 1.0)
        
        for label in axs[2].get_yticklabels():
            label.set_fontsize(14)  # Change the fontsize here
        # Add percentage on top of each bar in the second plot
        for bar in bars2:
            height = bar.get_height()
            height_pos = bar.get_height() + 0.004
            axs[2].text(bar.get_x() + bar.get_width() / 2, height_pos, f'{height*100:.1f}%', ha='center', va='bottom', fontsize=13)
        
        fig.suptitle(f'Rank percentile among all negative triplets', fontsize=20)
        
        # Adjust layout
        plt.tight_layout()
        save_dir = save_dir
        plt.savefig(save_dir+f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_rank_percentile_among_all_negative_triplets.png')
        plt.show()
            
    
#####-----run the script-----#####
def main():
    args = arg_parse()
    train(args) 

if __name__ == '__main__':
    main()