# libraries
library(dplyr)
library(readxl)

### necessary reference profiles preparation
# working directory
working_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/Oncodrug/"
# this directory is where you download your umls MRCONSO.RRF reference file
mrconso_dir = "/project/pi_rachel_melamed_uml_edu/Panos/drug_combo_jianfeng/CT_20250605/"

# read UMLS zip file
rrf = fread(paste0(mrconso_dir, "umls-2025AA-mrconso.zip"), sep = "|", quote = "")
rrf = rrf[, -19]
# Column names: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly
colnames(rrf) = c("CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE" , "STR", "SRL", "SUPPRESS", "CVF")
# Filter for english terms only
rrf = rrf %>% 
  filter(LAT == "ENG" & SAB == "MSH")
rrf$STR_lowercase = tolower(rrf$STR)
# Keep only necessary columns in rrf
rrf_sub <- rrf[, .(CODE, STR_lowercase)]

# drug bank vocabulary unzip csv file
db_file = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/final_result_tables_mapping/drugbank vocabulary.csv"
# read DrugBank csv zip file
db_csv <- read.csv(db_file, sep = ",", header = TRUE, stringsAsFactors = FALSE)

# manually created drug name to ID dictionary
manual_drug_name <- c("Azd4547", "Azd4320", "Mk-2206", "Bez-235", "Pd325901", 
                      "Abt-888", "Lapatinib ditosylate", "Azd-8186", "Zolinza", "Gemcitabine hydrochloride", 
                      "Pazopanib hydrochloride", "Sorafenib tosylate", "Erlotinib hydrochloride", "MK-1775", "Tozasertib", 
                      "SG3199", "AZD0156", "AZD4320", "AZD5991", "AZD5153", 
                      "AZD8186", "AZD1390", "AZD2811", "AZD7648", "AZD5363", 
                      "Ditc", "AZD6738", "AZD1775", "L744832", "ST1926", 
                      "PD98059")
manual_drug_code <- c("DB12247", NA, "DB16828", "DB11651", "DB07101", 
                      "DB07232", "DB01259", "DB15029", "DB02546", "DB00441", 
                      "DB06589", "DB00398", "DB00530", "DB11740", NA, 
                      NA, NA, NA, "DB14792", "DB17018", 
                      "DB15029", NA, NA, "DB16834", "DB12218", 
                      NA, "DB14917", "DB11740", NA, NA, 
                      NA)
drugname2code <- setNames(manual_drug_code, manual_drug_name)


# manually created condition name (indication) to ID dictionary
manual_condition_name <- c("metastatic breast cancer", "advanced thyroid carcinoma", "metastatic colorectal cancer", 
                           "metastatic urothelial carcinoma", "hairy-cell leukemia", "low-grade gliomas", 
                           "breast adenocarcinoma", "colorectal adenocarcinoma", "gastrointestinal neuroendocrine tumor", 
                           "her2-receptor positive breast cancer", "bowel cancer", "ampullary adenocarcinoma", 
                           "urothelial carcinoma", "gastroesophageal junction adenocarcinoma", "esophageal adenocarcinoma", 
                           "myeloproliferative neoplasms", "lymphatic cancer", "metastatic pancreatic cancer", 
                           "cns; brain cancer", "metastatic castration-resistant prostate cancer", "gastric cancer; colorectal cancer", 
                           "advanced esophagogastric adenocarcinoma", "esophageal; stomach cancer", "her2(-) advanced breast cancer", 
                           "advanced esophageal cancer", "advanced pancreatic cancer", "advanced anal cancer", 
                           "pleural cancer", "metastatic ovarian cancer", "desmoid tumours", 
                           "gastroesophageal cancer", "advanced breast cancer", "advanced ovarian cancer", 
                           "advanced gastrointestinal stromal tumors", "advanced pancreatic ductal adenocarcinoma", "advanced caecal cancer", 
                           "advanced biliary tract聽cancer", "metastatic esophageal cancer", "low-grade serous ovarian cancer", 
                           "inflammatory myofibroblastic tumor", "advanced rectal cancer", "metastatic bladder cancer", 
                           "peritoneal cancer", "advanced bile duct cancer", "multiple relapsed/refractory germ cell tumours", 
                           "for the the induction of remission in patients with acute promyelocytic leukemia", "estrogen receptor-negative breast cancer", "refractory multiple myeloma", 
                           "egfr-overexpressing breast cancer", "ovarian carcinoma", "adenocarcinoma of the gastroesophageal junction", 
                           "unresectable, metastatic biliary tract carcinoma", "metastatic gastric or gastroesophageal junction cancer", "androgen-independent prostate cancer", 
                           "estrogen receptor-positive breast cancer", "recurrent adult burkitt lymphoma", "cervical carcinoma", 
                           "mucosal melanoma", "parotid gland cancer", "pancreatic squamous cell carcinoma", 
                           "pancreatic ductal adenocarcinoma", "gastric adenocarcinoma", "pancreatic adenocarcinoma")
manual_condition_code <- c('D001943', 'D013964', 'D015179', 
                           'D001749', 'D007943', 'D005910', 
                           'D001943', 'D015179', 'D018358', 
                           'D001943', 'D003110', 'D000230', 
                           'D002295', 'D004938', 'C562730', 
                           'D009196', 'D008223', 'D010190', 
                           'D001932', 'D011471', 'D013274', 
                           'C562730', 'D004938', 'D001943', 
                           'D004938', 'D010190', 'D000694', 
                           'D010997', 'D010051', 'D018222', 
                           'D004938', 'D001943', 'D010051', 
                           'D046152', 'D021441', 'D002430', 
                           'D001661', 'D004938', 'D010051', 
                           'D047708', 'D015179', 'D001749', 
                           'D010534', 'D001650', 'D009373', 
                           'D015473', 'D001943', 'D009101', 
                           'D001943', 'D010051', 'D004938', 
                           'D001661', 'D004938', 'D064129', 
                           'D001943', 'D002051', 'D002583', 
                           'D008545', 'D010307', 'D010190', 
                           'D021441', 'D013274', 'D021441')
name2code <- setNames(manual_condition_code, manual_condition_name)


# manually created condition name (cancer type) to ID dictionary
# we mark Pancancer as NA as it's too general
manual_cancer_name <- c("Invasive Breast Carcinoma", "Bladder Urothelial Carcinoma", "Lung Neuroendocrine Tumor", 
                        "Esophagogastric Adenocarcinoma", "Colorectal Adenocarcinoma", "Ovarian Epithelial Tumor", 
                        "Unknown", "Prostate Adenocarcinoma", "Encapsulated Glioma", 
                        "Diffuse Glioma", "Lymphoid Neoplasm", "Myeloid Neoplasm", 
                        "Pancancer", "Breast sarcoma", "Lymphoid neoplasm", 
                        "Ovarian epithelial tumor", "Invasive breast carcinoma", "Colorectal adenocarcinoma", 
                        "Esophagogastric adenocarcinoma", "Ovary fallopian tube", "Pancreatic adenocarcinoma", 
                        "Bladder urinary tract", "Prostate adenocarcinoma", "Soft tissue", 
                        "Encapsulated glioma", "Bladder urothelial carcinoma", "Cervical squamous cell carcinoma", 
                        "Non-seminomatous germ cell tumor")
manual_cancer_code <- c("D001943", "D001749", "D018358", 
                        "D004938", "D015179", "D000077216", 
                        NA, "D011471", "D005910", 
                        "D005910", "D018190", "D019337", 
                        NA, "D001943", "D018190", 
                        "D000077216", "D001943", "D015179", 
                        "D004938", "D010049", "D021441", 
                        "D001743", "D011471", NA, 
                        "D005910", "D001749", "D002294", 
                        "D009373")
cancername2code <- setNames(manual_cancer_code, manual_cancer_name)

### mapping drug name to drug bank ID
# read the Oncodrug paper xlsx file
Onco_A <- read_excel(file.path(working_dir, "LevelA_dataset.xlsx"))
Onco_B <- read_excel(file.path(working_dir, "LevelB_dataset.xlsx"))
Onco_C <- read_excel(file.path(working_dir, "LevelC_dataset.xlsx"))
Onco_D <- read_excel(file.path(working_dir, "LevelD_dataset.xlsx"))

# Function to clean: remove empties and trim spaces
clean_parts <- function(x) {
  x <- trimws(x)        # remove leading/trailing spaces
  x <- x[x != ""]       # drop empty elements
  return(x)
}

### For A:
# Remove any "NA" strings
Onco_A$Drug_comb <- paste(Onco_A$`Targeted drug`, Onco_A$`Non-targeted drug`, sep=";")
Onco_A$Drug_comb_clean <- gsub("NA", "", Onco_A$Drug_comb)
# Split by ";"
splits <- strsplit(Onco_A$Drug_comb_clean, ";")
# Keep only rows with exactly 2 valid elements
valid_idx <- sapply(splits, function(x) length(clean_parts(x)) == 2)
# Initialize new columns
Onco_A$drug1_name <- NA
Onco_A$drug2_name <- NA
# Fill for valid rows
Onco_A$drug1_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[1])
Onco_A$drug2_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[2])
# Map drug1_name to DrugBank.ID
Onco_A <- Onco_A %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug1_name = Common.name, drug1_id = DrugBank.ID),
            by = "drug1_name") %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug2_name = Common.name, drug2_id = DrugBank.ID),
            by = "drug2_name")
# # find the missing mapping ones and manually map (No missing for A)
# Onco_A_missing <- Onco_A %>%
#   filter((!is.na(drug1_name) & is.na(drug1_id)) |
#            (!is.na(drug2_name) & is.na(drug2_id)))
### mapping mesh term to mesh code
Onco_A$cancer_type_name <- tolower(Onco_A$`Cancer type`)
Onco_A <- Onco_A %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(cancer_type_name = STR_lowercase, condition_id1 = CODE),
    by = "cancer_type_name"
  )
Onco_A$indication_name <- tolower(Onco_A$`Drug combination indications in sources`)
Onco_A <- Onco_A %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(indication_name = STR_lowercase, condition_id2 = CODE),
    by = "indication_name"
  )
Onco_A$condition_id <- ifelse(!is.na(Onco_A$condition_id1),
                              Onco_A$condition_id1,
                              Onco_A$condition_id2)
# Replace only if condition_id is NA
Onco_A$condition_id[is.na(Onco_A$condition_id)] <-
  name2code[Onco_A$indication_name[is.na(Onco_A$condition_id)]]
# # find missing match if any
# Onco_A_cond_missing <- Onco_A[which(is.na(Onco_A$condition_id)), ]
# # manually map dataframe with mapping info
# list1 = unique(Onco_A_cond_missing$indication_name)
# save the final table
Onco_A_clean <- Onco_A[, c("drug1_id", "drug2_id", "condition_id")] 
Onco_A_clean <- na.omit(Onco_A_clean)
write.csv(Onco_A_clean,
          file = file.path(working_dir, "Onco_A_clean.csv"),
          row.names = FALSE)


### For B:
Onco_B$Drug_comb <- paste(Onco_B$`Targeted drug`, Onco_B$`Non-targeted drug`, sep=";")
Onco_B$Drug_comb_clean <- gsub("NA", "", Onco_B$Drug_comb)
splits <- strsplit(Onco_B$Drug_comb_clean, ";")

# Keep only rows with exactly 2 valid elements
valid_idx <- sapply(splits, function(x) length(clean_parts(x)) == 2)
Onco_B$drug1_name <- NA
Onco_B$drug2_name <- NA
Onco_B$drug1_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[1])
Onco_B$drug2_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[2])

# Map drug1_name to DrugBank.ID
Onco_B <- Onco_B %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug1_name = Common.name, drug1_id = DrugBank.ID),
            by = "drug1_name") %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug2_name = Common.name, drug2_id = DrugBank.ID),
            by = "drug2_name")
# find the missing mapping ones and manually map
Onco_B_missing <- Onco_B %>%
  filter((!is.na(drug1_name) & is.na(drug1_id)) |
           (!is.na(drug2_name) & is.na(drug2_id)))
# manually mapping (find 5 among 8 in total, other three "drugs" are:)
# 'FOLFOXIRI', 'FOLFOX6', 'Chemotherapy'
Onco_B$drug2_id[which(Onco_B$drug2_name=='Furmonertinib')] <- 'DB16087'
Onco_B$drug2_id[which(Onco_B$drug2_name=='BNC105P')] <- 'DB06313'
Onco_B$drug2_id[which(Onco_B$drug2_name=='PLX9486')] <- 'DB18041'
Onco_B$drug2_id[which(Onco_B$drug2_name=='Trifluridine/Tipiracil')] <- 'DB16087'

### mapping mesh term to mesh code
Onco_B$cancer_type_name <- tolower(Onco_B$`Cancer type`)
Onco_B <- Onco_B %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(cancer_type_name = STR_lowercase, condition_id1 = CODE),
    by = "cancer_type_name"
  )
Onco_B$indication_name <- tolower(Onco_B$`Drug combination indications in sources`)
Onco_B <- Onco_B %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(indication_name = STR_lowercase, condition_id2 = CODE),
    by = "indication_name"
  )
Onco_B$condition_id <- ifelse(!is.na(Onco_B$condition_id1),
                              Onco_B$condition_id1,
                              Onco_B$condition_id2)
# find missing match
Onco_B_cond_missing <- Onco_B[which(is.na(Onco_B$condition_id)), ]
# manually map dataframe with mapping info
list1 = unique(Onco_B_cond_missing$indication_name)

# Replace only if condition_id is NA
Onco_B$condition_id[is.na(Onco_B$condition_id)] <-
  name2code[Onco_B$indication_name[is.na(Onco_B$condition_id)]]
# map NA to "cancer"
Onco_B$condition_id[which(Onco_B$condition_id=="D009369")] <- NA

# save the final table
Onco_B_clean <- Onco_B[, c("drug1_id", "drug2_id", "condition_id")] 
Onco_B_clean <- na.omit(Onco_B_clean)
write.csv(Onco_B_clean,
          file = file.path(working_dir, "Onco_B_clean.csv"),
          row.names = FALSE)


### For C:
Onco_C$Drug_comb <- paste(Onco_C$`Targeted drug`, Onco_C$`Non-targeted drug`, sep=";")
Onco_C$Drug_comb_clean <- gsub("NA", "", Onco_C$Drug_comb)
splits <- strsplit(Onco_C$Drug_comb_clean, ";")

# Keep only rows with exactly 2 valid elements
valid_idx <- sapply(splits, function(x) length(clean_parts(x)) == 2)
Onco_C$drug1_name <- NA
Onco_C$drug2_name <- NA
Onco_C$drug1_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[1])
Onco_C$drug2_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[2])

# Map drug1_name and drug2_name to DrugBank.ID
Onco_C <- Onco_C %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug1_name = Common.name, drug1_id = DrugBank.ID),
            by = "drug1_name") %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug2_name = Common.name, drug2_id = DrugBank.ID),
            by = "drug2_name")
# find the drugs with missing ID
Onco_C_missing <- Onco_C %>%
  filter((!is.na(drug1_name) & is.na(drug1_id)) |
           (!is.na(drug2_name) & is.na(drug2_id)))
# find the missing mapping ones and manually map
na_drug_names <- unique(c(
  Onco_C_missing$drug1_name[is.na(Onco_C_missing$drug1_id)],
  Onco_C_missing$drug2_name[is.na(Onco_C_missing$drug2_id)]
))
# Map the missing drug names to their codes
mapped_drug1_id <- drugname2code[Onco_C$drug1_name]
mapped_drug2_id <- drugname2code[Onco_C$drug2_name]
# Replace NAs in the original dataframe if a mapping exists
Onco_C$drug1_id[is.na(Onco_C$drug1_id)] <- mapped_drug1_id[is.na(Onco_C$drug1_id)]
Onco_C$drug2_id[is.na(Onco_C$drug2_id)] <- mapped_drug2_id[is.na(Onco_C$drug2_id)]

### mapping mesh term to mesh code
Onco_C$cancer_type_name <- tolower(Onco_C$`Cancer type`)
Onco_C <- Onco_C %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(cancer_type_name = STR_lowercase, condition_id1 = CODE),
    by = "cancer_type_name"
  )
Onco_C$indication_name <- tolower(Onco_C$`Drug combination indications in sources`)
Onco_C <- Onco_C %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(indication_name = STR_lowercase, condition_id2 = CODE),
    by = "indication_name"
  )
Onco_C$condition_id <- ifelse(!is.na(Onco_C$condition_id1),
                              Onco_C$condition_id1,
                              Onco_C$condition_id2)
# find missing match
Onco_C_cond_missing <- Onco_C[which(is.na(Onco_C$condition_id)), ]
# manually map dataframe with mapping info
list1 = unique(Onco_C_cond_missing$`Cancer type`)
# Replace only if condition_id is NA
Onco_C$condition_id[is.na(Onco_C$condition_id)] <-
  cancername2code[Onco_C$`Cancer type`[is.na(Onco_C$condition_id)]]
# map NA to "cancer"
Onco_C$condition_id[which(Onco_C$condition_id=="D009369")] <- NA

# save the final table
Onco_C_clean <- Onco_C[, c("drug1_id", "drug2_id", "condition_id")] 
Onco_C_clean <- na.omit(Onco_C_clean)
write.csv(Onco_C_clean,
          file = file.path(working_dir, "Onco_C_clean.csv"),
          row.names = FALSE)


### For D: (as there are too many triplets here, we don't do manually mapping) 
Onco_D$Drug_comb <- paste(Onco_D$`Targeted drug`, Onco_D$`Non-targeted drug`, sep=";")
Onco_D$Drug_comb_clean <- gsub("NA", "", Onco_D$Drug_comb)
splits <- strsplit(Onco_D$Drug_comb_clean, ";")

# Keep only rows with exactly 2 valid elements
valid_idx <- sapply(splits, function(x) length(clean_parts(x)) == 2)
Onco_D$drug1_name <- NA
Onco_D$drug2_name <- NA
Onco_D$drug1_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[1])
Onco_D$drug2_name[valid_idx] <- sapply(splits[valid_idx], function(x) clean_parts(x)[2])

# Map drug1_name and drug2_name to DrugBank.ID
Onco_D <- Onco_D %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug1_name = Common.name, drug1_id = DrugBank.ID),
            by = "drug1_name") %>%
  left_join(db_csv %>% dplyr::select(Common.name, DrugBank.ID) %>%
              rename(drug2_name = Common.name, drug2_id = DrugBank.ID),
            by = "drug2_name")

### mapping mesh term to mesh code
Onco_D$cancer_type_name <- tolower(Onco_D$`Cancer type`)
Onco_D <- Onco_D %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(cancer_type_name = STR_lowercase, condition_id1 = CODE),
    by = "cancer_type_name"
  )
Onco_D$indication_name <- tolower(Onco_D$`Drug combination indications in sources`)
Onco_D <- Onco_D %>%
  left_join(
    rrf_sub %>% dplyr::select(CODE, STR_lowercase) %>% 
      rename(indication_name = STR_lowercase, condition_id2 = CODE),
    by = "indication_name"
  )
Onco_D$condition_id <- ifelse(!is.na(Onco_D$condition_id1),
                              Onco_D$condition_id1,
                              Onco_D$condition_id2)
# find missing match
Onco_D_cond_missing <- Onco_D[which(is.na(Onco_D$condition_id)), ]
# manually map dataframe with mapping info
list1 = unique(Onco_D_cond_missing$`Cancer type`)
# Replace only if condition_id is NA
Onco_D$condition_id[is.na(Onco_D$condition_id)] <-
  cancername2code[Onco_D$`Cancer type`[is.na(Onco_D$condition_id)]]

# save the final table
Onco_D_clean <- Onco_D[, c("drug1_id", "drug2_id", "condition_id")] 
Onco_D_clean <- na.omit(Onco_D_clean)
write.csv(Onco_D_clean,
          file = file.path(working_dir, "Onco_D_clean.csv"),
          row.names = FALSE)






