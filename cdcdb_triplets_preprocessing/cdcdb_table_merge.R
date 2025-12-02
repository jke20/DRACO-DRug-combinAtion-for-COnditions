##### this file preprocess two tables downloaded from cdcdb into one merged file
##### before running this Rscript, please run cdcdb_file_download.ipynb first
### Goal: Map conditions to MeSH IDs 
### and with drug combinations (with at most 2 drugs) by nct_id

library(data.table)
library(dplyr)
library(jsonlite)
library(knitr)
library(httr)

# this is the directory where you save your two tables from cdcdb: drug_trial.csv and mesh_terms.csv
cdcdb_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/cdcdb_triplets_preprocessing/"
## load Jianfeng data files
drug_trial = fread(paste0(cdcdb_dir,"drug_trial.csv"))
trial_cond_mesh = fread(paste0(cdcdb_dir,"mesh_terms.csv"))

### Load MRCONSO.RRF (UMLS)
# this directory is where you download your umls MRCONSO.RRF reference file
mrconso_dir = "/project/pi_rachel_melamed_uml_edu/Panos/drug_combo_jianfeng/CT_20250605/"
rrf = fread(paste0(mrconso_dir, "umls-2025AA-mrconso.zip"), sep = "|", quote = "")
rrf = rrf[, -19]
# Column names meaning: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly
colnames(rrf) = c("CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE" , "STR", "SRL", "SUPPRESS", "CVF")
# Filter for english terms only
rrf = rrf %>% filter(LAT == "ENG" & SAB == "MSH")
rrf$STR_lowercase = tolower(rrf$STR)

## re-order columns in Jianfeng data files
drug_trial = drug_trial %>% 
  dplyr::select(drugbank_identifiers, nct_id = source_id) %>%
  distinct()
# keep only rows with max two drugs --> I do this by counting the number of "DB" appears
# If 2 times --> 2 drugs
# If >2 times --> >2 drugs
rows_to_keep = stringr::str_count(drug_trial$drugbank_identifiers, "DB") == 2
drug_trial = drug_trial[rows_to_keep, ] ; rownames(drug_trial) = NULL

trial_cond_mesh = trial_cond_mesh %>%
  dplyr::select(nct_id, mesh_terms_downcase) %>%
  distinct()

## match conditions to MeSH IDs
unique_conditions = trial_cond_mesh %>% 
  dplyr::select(mesh_terms_downcase) %>% 
  distinct() %>% # 2,173 unique conditions with MeSH terms
  left_join(rrf[, c(14, 19)], by = c("mesh_terms_downcase" = "STR_lowercase"))
colnames(unique_conditions) = c("MESH_TERM", "MESH_CODE")

# check how many didn't match to MeSH IDs
# sum(is.na(unique_conditions$CODE)) # 0 --> all terms matched to MeSH IDs!

## prepare files to save
trial_cond_mesh = left_join(trial_cond_mesh, unique_conditions, by = c("mesh_terms_downcase" = "MESH_TERM"))
colnames(trial_cond_mesh) = c("nct_id", "MESH_TERM_DOWNCASE", "MESH_CODE")
drug_trial = left_join(drug_trial, trial_cond_mesh, by = "nct_id")

# data.table::fwrite(trial_cond_mesh, paste0(cdcdb_dir, "trial_cond_MeSH_IDs.txt"), sep = "\t", row.names = FALSE)
data.table::fwrite(drug_trial, paste0(cdcdb_dir, "drugcombo_max2_drugs_conditions_MeSH_IDs.txt"), sep = "\t", row.names = FALSE)
# sum(is.na(drug_trial$MESH_TERM_DOWNCASE))
# sum(is.na(drug_trial$MESH_CODE))
