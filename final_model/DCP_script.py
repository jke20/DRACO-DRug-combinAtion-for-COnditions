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
                        train_partition = 1.0, 
                        test_partition = 0.0, 
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
    ct_dir = ct_dir
    # this is the directory where you have the cleaned drkg embeddings for drugs and conditions
    drkg_dir = drkg_dir
    # this is the directory where you save the drugs and conditions embeddings after training and finetuning
    emb_dir = emb_dir
    # this is the directory where you save the cdcdb triplets tables
    cdcdb_dir = cdcdb_dir
    # this is the directory where you plan to save the model running and evaluation results
    save_dir = save_dir

    
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
    ### triplets are already sorted here!!!
    all_comb_triplets = [
        sorted([drug1, drug2]) + [cond]
        for drug1, drug2, cond in zip(combined_df['drug1_idx'], combined_df['drug2_idx'], combined_df['cond_idx'])
    ]

    # All samples for training
    training_triplets = all_comb_triplets
    testing_triplets = []
    unique_drug_in_training = sorted(list(set(list(np.array(training_triplets).T[0]) + list(np.array(training_triplets).T[1]))))
    print(f'There are {len(training_triplets)} triplets in training with {len(unique_drug_in_training)} unique drugs.')

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
            # also sort existing triplets for the filtering step later
            existed_comb = [sorted(t[:2]) + [t[2]] for t in sub_pos_triplets]
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
            # other filters, if existed_comb is not sorted, there could be a leak of positive triplets in negative triplets
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

        return train_loss, train_roc_auc, train_pr_auc
    
    
    #####-----model training-----#####
    switch = False
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

        # positive pairs and testing positive pairs
        training_triplets = training_triplets

        # Initial error
        train_loss, train_roc_auc, train_pr_auc = test(multi)
        # append the result
        train_loss_list.append(train_loss)
        train_rocauc_list.append(train_roc_auc)
        train_prauc_list.append(train_pr_auc)
        print(f'Initial error:\n' + 
              f'Train Loss:{train_loss:.4f}, roc_auc:{train_roc_auc:.4f}, pr_auc:{train_pr_auc:.4f};\n')

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
                train_loss, train_roc_auc, train_pr_auc = test(multi)
                # append the result
                train_loss_list.append(train_loss)
                train_rocauc_list.append(train_roc_auc)
                train_prauc_list.append(train_pr_auc)

                print(f'Epoch:{(e+1):03d},\n' + 
                      f'Train Loss:{train_loss:.4f}, roc_auc:{train_roc_auc:.4f}, pr_auc:{train_pr_auc:.4f},\n')
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
        torch.save(model, save_dir+m_name)
        torch.save(model.state_dict(), save_dir+w_name)
        
        
        #####-----Plot-----#####
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(f"Loss and Accuracy curve for Training;\n" +
                     f"model name: {model_name}, lr: {lr}, negative samples:positive samples={multi}:1", fontsize=10)
    
        ax.plot(range(0, 1+epoch, 1), train_loss_list, color='black', label="Training Loss")
        ax.plot(range(0, 1+epoch, 1), train_rocauc_list, color='orange', label="ROC AUC")
        ax.plot(range(0, 1+epoch, 1), train_prauc_list, color='blue', label="PR AUC")
        ax.set_title('Training triplets')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_dir+f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_loss_accuracy_curve_after_finetuning.png')


    #####-----trained model loading-----#####
    # read the saved model
    model_name = model_name
    save_dir = save_dir
    m_name = f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_architecture_after_finetuning.pth'
    w_name = f'DCP_{model_name}_{lr}_{epoch}_{MLP_hl1_size}_{MLP_hl2_size}_{dropout}_{batch_size}_{multi}_{seed}_{train_partition}_{test_partition}_{sampling_ratio}_weights_after_finetuning.pth'
    final_model = torch.load(save_dir + m_name)
    final_model.load_state_dict(torch.load(save_dir + w_name))
    final_model.eval()

    
    #####-----propose top 3 triplets for each condition-----#####
    #####-----find all predictions passing the threshold 0.1-----#####
    switch = True
    if switch:
        # find all the drug idx and condition idx
        all_unique_drug_idx_list = sorted(set(drug_cond_df['drug_idx'].tolist()))
        all_unique_cond_idx_list = sorted(set(drug_cond_df['cond_idx'].tolist()))
    
        # set a threshold
        pred_thred = 0.1
        # these are the predicted triplets that pass the threshold
        threshold_triplets, threshold_triplets_scores = [], []
        # List to store top 3 triplets and their scores for each condition
        top_3_triplets, top_3_triplets_scores = [], []
        
        # Convert all_comb_triplets to a set for faster membership checking
        all_comb_triplets_set = set(map(tuple, all_comb_triplets))
        # record the time
        start_time = time.time()
        
        # iteration to find predictions with predicted scores greater than the threshold
        for cond_idx in all_unique_cond_idx_list:
            tmp_triplets1 = [[x, y, cond_idx] for x, y in itertools.combinations(all_unique_drug_idx_list, 2)]
            # filter out existing triplets
            triplets = [triplet for triplet in tmp_triplets1 if tuple(triplet) not in all_comb_triplets_set]
            
            # run the model
            drug_list1, drug_list2, cond_list = np.array(triplets).T
            pred = final_model.forward(drug_list1, drug_list2, cond_list)
            pred = np.array(pred.detach().numpy())
    
            filtered_triplets = [(triplet, p) for triplet, p in zip(triplets, pred) if p > pred_thred]
            if filtered_triplets:
                threshold_triplets_pass, threshold_triplets_scores_pass = zip(*filtered_triplets)
                threshold_triplets_pass = list(threshold_triplets_pass)
                threshold_triplets_scores_pass = list(threshold_triplets_scores_pass)
                
                # Sort before appending
                sorted_pass_pairs = sorted(zip(threshold_triplets_pass, threshold_triplets_scores_pass), key=lambda x: x[1], reverse=True)
                threshold_triplets_pass, threshold_triplets_scores_pass = zip(*sorted_pass_pairs)
                threshold_triplets_pass = list(threshold_triplets_pass)
                threshold_triplets_scores_pass = list(threshold_triplets_scores_pass)
                
                # Append the results
                threshold_triplets = threshold_triplets + threshold_triplets_pass
                threshold_triplets_scores = threshold_triplets_scores + threshold_triplets_scores_pass
            
                # Sort triplets by predicted scores in descending order and get the top 3
                top_3_triplets_for_cond = sorted(zip(threshold_triplets_pass, threshold_triplets_scores_pass),
                                                 key=lambda x: x[1], reverse=True)[:3]
                
                # Record the top 3 triplets and their scores for the current condition
                for triplet, score in top_3_triplets_for_cond:
                    top_3_triplets.append(triplet)
                    top_3_triplets_scores.append(score)
    
            # print the running time
            now_time = time.time()
            total_time = now_time - start_time
            print('Running time: {:.2f} seconds'.format(total_time))
    
        ### map the idx to the names
        # top 3 triplets matrix
        drug1_list, drug2_list, cond_list = np.array(top_3_triplets).T
        data = {
            'drug1_idx': drug1_list,
            'drug2_idx': drug2_list,
            'cond_idx': cond_list,
            'scores': top_3_triplets_scores
        }
        df = pd.DataFrame(data)
        df['drug1'] = [name for idx in drug1_list for name, value in emb_id_mapping.items() if idx==value]
        df['drug2'] = [name for idx in drug2_list for name, value in emb_id_mapping.items() if idx==value]
        df['condition'] = [name for idx in cond_list for name, value in emb_id_mapping.items() if idx==value]
        top_3_df = df[['drug1', 'drug2', 'condition', 'scores']]
        
        # triplets that pass the threshold
        drug1_list, drug2_list, cond_list = np.array(threshold_triplets).T
        data = {
            'drug1_idx': drug1_list,
            'drug2_idx': drug2_list,
            'cond_idx': cond_list,
            'scores': threshold_triplets_scores
        }
        df = pd.DataFrame(data)
        df['drug1'] = [name for idx in drug1_list for name, value in emb_id_mapping.items() if idx==value]
        df['drug2'] = [name for idx in drug2_list for name, value in emb_id_mapping.items() if idx==value]
        df['condition'] = [name for idx in cond_list for name, value in emb_id_mapping.items() if idx==value]
        pass_threshold_df = df[['drug1', 'drug2', 'condition', 'scores']]

        filename1 = 'top_3_triplets_per_condition.csv'
        top_3_df.to_csv(save_dir+filename1)
        filename2 = 'triplets_pass_threshold.csv'
        pass_threshold_df.to_csv(save_dir+filename2)    


    #####-----propose top 3 triplets for each condition-----#####
    switch = True
    if switch:
        # find all the drug idx and condition idx
        all_unique_drug_idx_list = sorted(set(drug_cond_df['drug_idx'].tolist()))
        all_unique_cond_idx_list = sorted(set(drug_cond_df['cond_idx'].tolist()))
        
        def find_second_drug(drug_idx, cond_idx, threshold):
            df_sub = combined_df[combined_df['cond_idx']==cond_idx]
            drug_list = list(df_sub['drug1']) + list(df_sub['drug2'])
            top_preds = []
            # if this drug never present in any drug combination
            if len(drug_list)==0 or drug_idx not in drug_list:
                other_drug_list = [i for i in all_unique_drug_idx_list if i!=drug_idx]
                drug_idx_list = list(np.repeat(drug_idx, len(other_drug_list)))
                cond_idx_list = list(np.repeat(cond_idx, len(other_drug_list)))
                pred = final_model.forward(other_drug_list, drug_idx_list, cond_idx_list)
                pred_list = np.array(pred.detach().numpy())
                # we find the candidate second drugs
                # find the top 3 predictions
                top3_indices = np.argsort(pred_list)[-3:][::-1]
                for idx in top3_indices:
                    pred_score = pred_list[idx]
                    if pred_score > 0.1:
                        candidate_drug_idx = other_drug_list[idx]
                        top_preds.append((drug_idx, candidate_drug_idx, cond_idx, pred_score))
            
            return top_preds
        
        threshold = 0.1
        count = 0
        second_drug_propose_list = []
        for i in range(drug_cond_df.shape[0]):
            drug_idx = drug_cond_df['drug_idx'][i]
            cond_idx = drug_cond_df['cond_idx'][i]
            results = find_second_drug(drug_idx, cond_idx, threshold)
            if len(results)!=0:
                count += 1
                for r in results:
                    second_drug_propose_list.append(list(r))
        
        print(f"Among {drug_cond_df.shape[0]} clinical trials:")
        print(f"  - {count} of them don't exist in any cdcdb triplets")
        
        drug1_list, drug2_list, cond_list, score_list = np.array(second_drug_propose_list).T
        data = {
            'drug1_idx': drug1_list,
            'drug2_idx': drug2_list,
            'cond_idx': cond_list,
            'scores': score_list
        }
        df = pd.DataFrame(data)
        df['drug1'] = [name for idx in drug1_list for name, value in emb_id_mapping.items() if idx==value]
        df['drug2'] = [name for idx in drug2_list for name, value in emb_id_mapping.items() if idx==value]
        df['condition'] = [name for idx in cond_list for name, value in emb_id_mapping.items() if idx==value]
        second_drug_df = df[['drug1', 'drug2', 'condition', 'scores']]
        
        filename3 = 'clinical_trials_with_proposed_second_drugs.csv'
        second_drug_df.to_csv(save_dir+filename3)


#####-----run the script-----#####
def main():
    args = arg_parse()
    train(args) 

if __name__ == '__main__':
    main()
    
