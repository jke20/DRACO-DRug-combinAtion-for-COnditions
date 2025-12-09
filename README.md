# DRACO-DRug-combinAtion-for-COnditions
Code and data for the paper: A Knowledge Graph Approach To Discovering Drug Combination Therapies Across The Phenome
#### Full version of Table 1 & Supplementary tables 1-3 are located in folder: final_result_tables_mapping
- Table 1: Top50 unannotated cdcdb drug combinations.xlsx
- Supplementary tables 1: clinical_trials_with_proposed_second_drugs_w_names.csv
- Supplementary tables 2: top_3_triplets_per_condition_w_names.csv
- Supplementary tables 3: triplets_pass_threshold_w_names.csv
#### Note:
- Because users may store files in different locations, the provided files allow flexible directory settings. If you want all files to be stored in one folder, you can define multiple directories as the same path.
---

## Workflow

### **Step 1: Data Preprocessing and Preparation**
1. Download and preprocess clinical trial data.  
2. Download and preprocess DRKG embeddings.  
3. Download and preprocess CDCDB triplets (drug combinations + condition).

---

### **Step 2: Embedding Training and Finetuning**
1. Train embeddings for drugs and conditions that do not already have DRKG embeddings.  
2. Finetune embeddings for all nodes.

---

### **Step 3: Model Training and Evaluation**
1. Split all triplets into:  
   - 80% training   (for model training)
   - 10% testing    (for hyperparameters selection)
   - 10% validation (for model validation)

---

### **Step 4: Final Model Evaluation**
1. Retrain the final model using all available triplets.  
2. Evaluate the final model on:  
   - drug combinations without known conditions from CDCDB  
   - triplets from the OncoDrug+ database
