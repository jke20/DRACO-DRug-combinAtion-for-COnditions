### preprocessing and cleaning CT (clinical trial) data

library(data.table)
library(dplyr)
library(jsonlite)
library(knitr)
library(httr)

### Load clinical trials
# - Filter for interventional studies only
# - Filter for clinical trials with interventions of {biological, combination product, drug, genetic}
# - Filter for those that have mesh_terms mapped to both interventions and conditions

### directories
# this is the directory where you download and unzip clinical trials downloaded from AACT
# https://aact.ctti-clinicaltrials.org/downloads
aact_dir = "/project/pi_rachel_melamed_uml_edu/ref_data/AACT/20250605/"
# this is the directory where you want to save your preprocessed drug-condition pairs and triplets tables
ct_dir = "/project/pi_rachel_melamed_uml_edu/Jianfeng/Drug_combinations/06122025/clinical_trail_preprocessing/"
# this directory is where you download your umls MRCONSO.RRF reference file
mrconso_dir = "/project/pi_rachel_melamed_uml_edu/Panos/drug_combo_jianfeng/CT_20250605/"
# this directory is where your drugbank vocabulary for mapping drugbank drugs to RxNORM codes
db_vocal_dir = "/project/pi_rachel_melamed_uml_edu/ref_data/drugbank/20250606/"
# this is the apiKey for running UMLS API crosswalk (easily register)
apiKey = "8881fe0b-f54d-44b4-a248-984d6de51c5f"

### read the downloaded tables from AACT
ct_studies = fread(paste0(aact_dir, "studies.txt"))
# Filter for interventional studies
ct_studies_interventional = ct_studies %>% 
  filter(study_type == "INTERVENTIONAL")
# Filter for interventions of biological, combination product, drug, genetic
ct_interventions = fread(paste0(aact_dir, "interventions.txt"))
ct_interventions_filt = ct_interventions %>% 
  filter(intervention_type %in% c("BIOLOGICAL", "COMBINATION_PRODUCT", "DRUG", "GENETIC"))
ct_studies_interventional_filt = ct_studies_interventional %>% 
  filter(nct_id %in% ct_interventions_filt$nct_id) %>% 
  dplyr::select(nct_id, overall_status, last_known_status, phase)
# Filter for cts with mesh_terms to both interventions and conditions
ct_interventions_mesh = fread(paste0(aact_dir, "browse_interventions.txt"))
ct_conditions_mesh = fread(paste0(aact_dir, "browse_conditions.txt"))
ct_studies_interventional_filt = ct_studies_interventional_filt %>% 
  filter(nct_id %in% ct_interventions_mesh$nct_id & nct_id %in% ct_conditions_mesh$nct_id)
ct_interventions_mesh = ct_interventions_mesh %>%
  filter(nct_id %in% ct_studies_interventional_filt$nct_id) %>% 
  dplyr::select(-id)
ct_conditions_mesh = ct_conditions_mesh %>%
  filter(nct_id %in% ct_studies_interventional_filt$nct_id) %>%
  dplyr::select(-id)
rm(ct_studies, ct_studies_interventional, ct_interventions, ct_interventions_filt)
# Keep only mesh-list
ct_conditions_mesh = ct_conditions_mesh %>% 
  filter(mesh_type == "mesh-list") %>%
  dplyr::select(nct_id, downcase_mesh_term) %>% distinct()
ct_interventions_mesh = ct_interventions_mesh %>%
  filter(mesh_type == "mesh-list") %>%
  dplyr::select(nct_id, downcase_mesh_term) %>% distinct()
ct_studies_interventional_filt = ct_studies_interventional_filt %>%
  left_join(ct_conditions_mesh, by = "nct_id") %>% 
  left_join(ct_interventions_mesh, by = "nct_id") %>%
  dplyr::rename(condition_downcase_mesh = downcase_mesh_term.x, intervention_downcase_mesh = downcase_mesh_term.y)
ct_conditions_unique = ct_conditions_mesh %>% 
  dplyr::select(downcase_mesh_term) %>% 
  distinct() %>% 
  arrange(downcase_mesh_term)
ct_interventions_unique = ct_interventions_mesh %>% 
  dplyr::select(downcase_mesh_term) %>% 
  distinct() %>% 
  arrange(downcase_mesh_term)
rm(ct_conditions_mesh, ct_interventions_mesh)

#### Clinical trial conditions ####
# Add MeSH codes
rrf = fread(paste0(mrconso_dir, "umls-2025AA-mrconso.zip"), sep = "|", quote = "")
rrf = rrf[, -19]
# Column names: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly
colnames(rrf) = c("CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE" , "STR", "SRL", "SUPPRESS", "CVF")
# Filter for english terms only
rrf = rrf %>% 
  filter(LAT == "ENG" & SAB == "MSH")
rrf$STR_lowercase = tolower(rrf$STR)
# For each ct_condition, add the corresponding mesh_code from the rrf
ct_conditions_unique = ct_conditions_unique %>%
  left_join(rrf[, c("STR_lowercase", "CODE")], by = c("downcase_mesh_term" = "STR_lowercase")) %>%
  dplyr::rename(mesh_code = CODE)
sum(is.na(ct_conditions_unique$mesh_code)) # 0 NAs introduced --> 100% match

### Clinical trial interventions 
# MSH term --> MSH code --> RxNORM

# MSH term -- > MSH codes
ct_interventions_unique = ct_interventions_unique %>%
  left_join(rrf[, c("STR_lowercase", "CODE")], by = c("downcase_mesh_term" = "STR_lowercase")) %>%
  dplyr::rename(mesh_code = CODE) %>% 
  distinct()
sum(is.na(ct_interventions_unique$mesh_code)) # 0 NAs introduced --> 100% match

# MSH codes --> RxNORM (UMLS API crosswalk)
ct_interventions_mesh_rxcui = data.frame(mesh_term = NA, mesh_code = NA, rxnorm_term = NA, rxcui = NA)
pageSize = 10000
for (i in 1:nrow(ct_interventions_unique)) {
  # Prepare UMLS API call
  source_vobaculary = "MSH"
  target_vocabulary = "RXNORM"
  mesh_code = ct_interventions_unique[i, "mesh_code"]
  call = paste0("https://uts-ws.nlm.nih.gov/rest/crosswalk/current/source/", source_vobaculary, "/", mesh_code, "?targetSource=", target_vocabulary ,"&apiKey=", apiKey, "&pageSize=", pageSize)
  # Call UMLS API
  res = GET(call)
  # res is a JSON output. Handle JSON
  data = fromJSON(rawToChar(res$content))
  # Get MDR codes and names
  rxcui = data$result$ui
  rxterm = data$result$name
  
  # If no rxcui returned
  if (length(rxcui) == 0) {
    no_match = data.frame(mesh_term = ct_interventions_unique[i, downcase_mesh_term], mesh_code = mesh_code, rxnorm_term = NA, rxcui = NA)
    ct_interventions_mesh_rxcui = rbind(ct_interventions_mesh_rxcui, no_match)
  }
  
  # If only 1 rxcui returned
  if (length(rxcui) == 1) { 
    match = data.frame(mesh_term = ct_interventions_unique[i, downcase_mesh_term], mesh_code = mesh_code, rxnorm_term = rxterm, rxcui = rxcui)
    ct_interventions_mesh_rxcui = rbind(ct_interventions_mesh_rxcui, match)
  }
  
  # If >1 rxcui returned
  if (length(rxcui) > 1) {
    for (z in 1:length(rxcui)) {
      match = data.frame(mesh_term = ct_interventions_unique[i, downcase_mesh_term], mesh_code = mesh_code, rxnorm_term = rxterm[z], rxcui = rxcui[z])
      # rbind to output data frame
      ct_interventions_mesh_rxcui = rbind(ct_interventions_mesh_rxcui, match)
    }
  }
  
  # Track progress
  cat(i, "/", nrow(ct_interventions_unique), "\n")
} ; rm(apiKey, call, i, mesh_code, pageSize, rxcui, rxterm, source_vobaculary, target_vocabulary, z, data, res, match, no_match)
ct_interventions_mesh_rxcui = ct_interventions_mesh_rxcui[-1, ]
rownames(ct_interventions_mesh_rxcui) = NULL

# Didn't match
interventions_mesh_to_rxcui_no_match = ct_interventions_mesh_rxcui[which(is.na(ct_interventions_mesh_rxcui$rxcui)), ]
length(unique(interventions_mesh_to_rxcui_no_match$mesh_code))
interventions_mesh_to_rxcui_no_match = interventions_mesh_to_rxcui_no_match[, 1:2]
colnames(interventions_mesh_to_rxcui_no_match) = c("intervention_mesh_term", "intervention_mesh_code")
# keep unmatched drug combinations and manually match them to drugbank IDs
mesh_to_drugbank_manually = interventions_mesh_to_rxcui_no_match[grepl("combination", interventions_mesh_to_rxcui_no_match$intervention_mesh_term),]
mesh_to_drugbank_manually$DB_drug1 = c("DB00613","DB00681",NA,"DB11751","DB00878","DB16393","DB04839","DB01288",NA,"DB00071","DB00030","DB16691","DB16474","DB12975")
mesh_to_drugbank_manually$DB_drug2 = c("DB09274","DB03619",NA,"DB08864","DB02513","DB16394","DB00977","DB00332",NA,"DB00030","DB00046","DB00503","DB16485","DB09274")
mesh_to_drugbank_manually = na.omit(mesh_to_drugbank_manually)
rownames(mesh_to_drugbank_manually) = NULL
# fwrite(mesh_to_drugbank_manually, paste0(mrconso_dir, "dcombo_MeSh_to_DrugBank_manual.txt"), sep = "\t", row.names = FALSE)
rm(interventions_mesh_to_rxcui_no_match)
# Remove them
ct_interventions_mesh_rxcui = na.omit(ct_interventions_mesh_rxcui)
rownames(ct_interventions_mesh_rxcui) = NULL
length(unique(ct_interventions_mesh_rxcui$mesh_code))
head(ct_interventions_mesh_rxcui)
colnames(ct_interventions_mesh_rxcui) = c("intervention_mesh_term", "intervention_mesh_code", "intervention_rxnorm_term", "intervention_rxnorm_code")
# fwrite(ct_interventions_mesh_rxcui, paste0(mrconso_dir, "[interventions]_mesh_to_rxnorm.txt"), sep = "\t", row.names = FALSE)

### Keep only those that are ingredients
rxcuis = unique(ct_interventions_mesh_rxcui$intervention_rxnorm_code)
rxcuis_attributes = data.frame(rela = NA, relaSource = NA, minConcept.rxcui = NA, minConcept.name = NA, minConcept.tty = NA, rxclassMinConceptItem.classId = NA, rxclassMinConceptItem.className = NA, rxclassMinConceptItem.classType = NA)
# For each drug: add all the information stored in RxNorm (including indications)
# Load output:
for (i in 1:length(rxcuis)) {
  # Prepare API call
  rxcui = rxcuis[i]
  call = paste0("https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui?rxcui=", rxcui)
  # call = paste0("https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName?drugName=", drug_name_http)
  # Call API
  get_cui = GET(url = call)
  # Converting content to text
  get_cui = content(get_cui, "text", encoding = "UTF-8")
  # Parsing data in JSON
  get_cui_json = fromJSON(get_cui, flatten = TRUE)
  # Converting into dataframe
  get_cui_json_dataframe = get_cui_json[["rxclassDrugInfoList"]][["rxclassDrugInfo"]]
  
  # If no match
  if (is.null(get_cui_json_dataframe) == TRUE) {
    get_cui_json_dataframe = get_cui_json_dataframe[, 1:8]
    null = data.frame(rela = NA, relaSource = NA, minConcept.rxcui = rxcui, minConcept.name = NA, minConcept.tty = NA, rxclassMinConceptItem.classId = NA,
                      rxclassMinConceptItem.className = NA, rxclassMinConceptItem.classType = NA)
    rxcuis_attributes = rbind(rxcuis_attributes, null)
  }
  
  # If match
  if (is.null(get_cui_json_dataframe) == FALSE) {
    get_cui_json_dataframe = get_cui_json_dataframe[, 1:8]
    rxcuis_attributes = rbind(rxcuis_attributes, get_cui_json_dataframe)
  }
  
  # Tracking progress
  cat(i, "/", length(rxcuis), "\n")
} ; rm(i, call, get_cui, rxcui, get_cui_json, get_cui_json_dataframe, null)
rxcuis_attributes = rxcuis_attributes[-1, ]
rownames(rxcuis_attributes) = NULL
# fwrite(rxcuis_attributes, paste0(mrconso_dir, "[interventions]_rxnorm_attributes.txt"), sep = "\t", row.names = FALSE)
rxcuis_attributes = rxcuis_attributes %>% 
  filter(minConcept.tty == "IN") %>%
  dplyr::select(rxcui = minConcept.rxcui, rxnorm_name = minConcept.name) %>%
  distinct()

## Filter rxcuis matched for only ingredients
ct_interventions_mesh_rxcui_ingredients = ct_interventions_mesh_rxcui %>% 
  filter(intervention_rxnorm_code %in% rxcuis_attributes$rxcui)
# fwrite(ct_interventions_mesh_rxcui_ingredients, paste0(mrconso_dir, "[interventions]_mesh_to_rxnorm_ingredients.txt"), sep = "\t", row.names = FALSE)

### Drug combinations
rxcuis = ct_interventions_mesh_rxcui %>% 
  dplyr::select(intervention_rxnorm_code) %>% 
  distinct()
colnames(rxcuis) = "rxcui"
# Match rxnorm to part_of
pageSize = 10000
rxnorm_combinations = data.frame(rxcui = NA, relation = NA, rxcui_new = NA, rxname_per_drug = NA)
for (i in 1:nrow(rxcuis)) {
  rxcui = rxcuis[i, "rxcui"]
  call = paste0("https://uts-ws.nlm.nih.gov/rest/content/current/source/RXNORM/", rxcui, "/relations?includeAdditionalRelationLabels=has_part&apiKey=", apiKey)
  # Call UMLS API
  res = GET(call)
  # res is a JSON output. Handle JSON
  data = fromJSON(rawToChar(res$content))
  
  if (length(data) == 4) {
    relation = data[["result"]][["additionalRelationLabel"]]
    rxcui_new = data[["result"]][["relatedId"]]
    rxname_per_drug = data[["result"]][["relatedIdName"]]
    if (length(relation) == 1) {
      y = data.frame(rxcui = rxcuis[i, "rxcui"],
                     relation = relation,
                     rxcui_new = basename(rxcui_new),
                     rxname_per_drug = rxname_per_drug)
      rxnorm_combinations = rbind(rxnorm_combinations, y)
    }
    if (length(relation) > 1) {
      for (a in 1:length(relation)) {
        y = data.frame(rxcui = rxcuis[i, "rxcui"],
                       relation = relation[a],
                       rxcui_new = basename(rxcui_new[a]),
                       rxname_per_drug = rxname_per_drug[a])
        rxnorm_combinations = rbind(rxnorm_combinations, y)
      }
    }
  }
  
  if (length(data) == 3) {
    y = data.frame(rxcui = rxcuis[i, "rxcui"],
                   relation = NA, 
                   rxcui_new = NA, 
                   rxname_per_drug = NA)
    rxnorm_combinations = rbind(rxnorm_combinations, y)
  }
  
  cat(i, "/", nrow(rxcuis), "\n")
} ; rm(apiKey, call, i, pageSize, rxcui, a, data, res, y, relation, rxname_per_drug, rxcui_new, rxcuis)
rxnorm_combinations = rxnorm_combinations[-1, ]
rownames(rxnorm_combinations) = NULL
# Remove non drug combinations
rxnorm_combinations = na.omit(rxnorm_combinations)
rownames(rxnorm_combinations) = NULL
# Check lengths
length(unique(rxnorm_combinations$rxcui)) # 64 drug combinations
colnames(rxnorm_combinations) = c("intervention_rxnorm_code", "relation", "rxnorm_code_per_drug", "rxnorm_term_per_drug")
# fwrite(rxnorm_combinations, paste0(mrconso_dir, "[interventions]_drug_combinations_rxnorm.txt"), sep = "\t", row.names = FALSE)

### Add to each file the nct_id
ct_studies_conditions = ct_studies_interventional_filt %>% 
  dplyr::select(nct_id, condition_downcase_mesh) %>% 
  distinct() %>%
  left_join(ct_conditions_unique, by = c("condition_downcase_mesh" = "downcase_mesh_term"))

ct_studies_interventions_ingredients = ct_studies_interventional_filt %>% 
  dplyr::select(nct_id, intervention_downcase_mesh) %>% 
  distinct() %>%
  left_join(ct_interventions_mesh_rxcui, by = c("intervention_downcase_mesh" = "intervention_mesh_term")) %>%
  filter(intervention_rxnorm_code %in% rxcuis_attributes$rxcui) %>%
  na.omit() %>%
  dplyr::select(nct_id, intervention_rxnorm_term, intervention_rxnorm_code) %>%
  distinct() %>%
  left_join(ct_studies_conditions, by = "nct_id") %>%
  na.omit() %>%
  distinct() %>%
  dplyr::select(intervention_rxnorm_term, intervention_rxnorm_code, mesh_code) %>%
  distinct()

## create file for drug combinations
ct_studies_interventions_dcombo = ct_studies_interventional_filt %>% 
  dplyr::select(nct_id, intervention_downcase_mesh) %>% 
  distinct() %>%
  left_join(ct_interventions_mesh_rxcui, by = c("intervention_downcase_mesh" = "intervention_mesh_term")) %>%
  filter(intervention_rxnorm_code %in% rxnorm_combinations$intervention_rxnorm_code) %>%
  na.omit() %>%
  left_join(rxnorm_combinations[,c("intervention_rxnorm_code", "rxnorm_code_per_drug", "rxnorm_term_per_drug")], by = "intervention_rxnorm_code") %>%
  dplyr::select(nct_id, intervention_rxnorm_code, rxnorm_code_per_drug, rxnorm_term_per_drug) %>%
  distinct() %>%
  left_join(ct_studies_conditions, by = "nct_id") %>%
  na.omit() %>%
  distinct() %>%
  dplyr::select(intervention_rxnorm_code_combination = intervention_rxnorm_code, rxnorm_code_per_drug, rxnorm_term_per_drug, mesh_code) %>%
  distinct()

## do the same for the manually matched drug combinations
ct_studies_manual_dcombo = ct_studies_interventional_filt %>% 
  dplyr::select(nct_id, intervention_downcase_mesh) %>% 
  distinct() %>%
  left_join(mesh_to_drugbank_manually, by = c("intervention_downcase_mesh" = "intervention_mesh_term")) %>%
  na.omit() %>%
  distinct() %>%
  left_join(ct_studies_conditions, by = "nct_id") %>%
  na.omit() %>%
  distinct() %>%
  dplyr::select(DB_drug1, DB_drug2, mesh_code) %>%
  distinct()

########################################################################
########################################################################
########################################################################

### Match all drugbank drugs to RxNORM codes
db_drugs = fread(paste0(db_vocal_dir, "drugbank_all_drugbank_vocabulary.csv.zip"))
db_drugs = db_drugs %>% 
  dplyr::select(drugbank_id = "DrugBank ID", drug_name = "Common name") %>%
  distinct()

rxnorm_drugbank_indications = data.frame(drugbank_id = NA, drugbank_name = NA, minConcept.rxcui = NA, minConcept.name = NA, minConcept.tty = NA)
# For each drug: add all the information stored in RxNorm (including indications)
### some Bad Request - URLDecoder: Incom may happen, need to manually continue the for loop
### May need to skip 9542, 9543
for (i in 1:nrow(db_drugs)) {
  # Prepare API call
  drug_db_id = db_drugs[i, drugbank_id]
  drug_name = db_drugs[i, drug_name]
  drug_name_http = gsub(" ", "%20", drug_name)
  call = paste0("https://rxnav.nlm.nih.gov/REST/rxclass/class/byDrugName?drugName=", drug_name_http)
  # Call API
  get_cui = GET(url = call)
  # Converting content to text
  get_cui = content(get_cui, "text", encoding = "UTF-8")
  # Parsing data in JSON
  get_cui_json = fromJSON(get_cui, flatten = TRUE)
  # Converting into dataframe
  get_cui_json_dataframe = get_cui_json[["rxclassDrugInfoList"]][["rxclassDrugInfo"]]
  
  # If no match
  if (is.null(get_cui_json_dataframe) == TRUE) {
    null = data.frame(drugbank_id = drug_db_id, drugbank_name = drug_name, minConcept.rxcui = NA, minConcept.name = NA, minConcept.tty = NA)
    rxnorm_drugbank_indications = rbind(rxnorm_drugbank_indications, null)
  }
  
  # If match
  if (is.null(get_cui_json_dataframe) != TRUE) {
    get_cui_json_dataframe = get_cui_json_dataframe %>% 
      dplyr::select(minConcept.rxcui, minConcept.name, minConcept.tty) %>% 
      distinct() %>%
      mutate(drugbank_id = all_of(drug_db_id),
             drugbank_name = all_of(drug_name), .before = minConcept.rxcui)
    rxnorm_drugbank_indications = rbind(rxnorm_drugbank_indications, get_cui_json_dataframe)
  }
  
  # Tracking progress
  cat(i, "/", nrow(db_drugs), "\n")
} ; rm(drug_name, drug_db_id, drug_name_http, call, get_cui, i, get_cui_json, get_cui_json_dataframe, null)
rxnorm_drugbank_indications = rxnorm_drugbank_indications[-1, ]

not_matched = rxnorm_drugbank_indications[which(is.na(rxnorm_drugbank_indications$minConcept.tty)), ]
rownames(not_matched) = NULL
not_matched = not_matched %>% 
  dplyr::select(drugbank_id, drugbank_name) %>% 
  distinct()

rxnorm_drugbank_indications_in = rxnorm_drugbank_indications %>%
  filter(minConcept.tty == "IN") %>% 
  distinct()

# Get PINs
x = c(rxnorm_drugbank_indications_in$drugbank_id, not_matched$drugbank_id)
pins= setdiff(db_drugs$drugbank_id, x)
rxnorm_drugbank_indications_pin = rxnorm_drugbank_indications %>%
  filter(drugbank_id %in% pins) %>%
  filter(minConcept.tty == "PIN") %>% 
  distinct()
rm(pins)

# Combine INs and PINs
rxnorm_drugbank_indications_in_pin = rbind(rxnorm_drugbank_indications_in, rxnorm_drugbank_indications_pin)
rm(rxnorm_drugbank_indications_in, rxnorm_drugbank_indications_pin)
rxnorm_drugbank_indications_in_pin = rxnorm_drugbank_indications_in_pin %>%
  dplyr::select(drugbank_id, rxnorm_cui = minConcept.rxcui) %>%
  distinct()

#### Create final files ####

## Convert RxNORM CUIs to DrugBank IDs
ct_studies_interventions_ingredients = ct_studies_interventions_ingredients %>%
  left_join(rxnorm_drugbank_indications_in_pin, by = c("intervention_rxnorm_code" = "rxnorm_cui")) %>%
  na.omit() %>%
  dplyr::select(drug = drugbank_id, condition = mesh_code) %>%
  distinct()

ct_studies_interventions_dcombo = ct_studies_interventions_dcombo %>% 
  left_join(rxnorm_drugbank_indications_in_pin, by = c("rxnorm_code_per_drug" = "rxnorm_cui")) %>% 
  dplyr::select(drugbank_id, intervention_rxnorm_code_combination, rxnorm_code_per_drug, mesh_code) %>%
  distinct()
# remove combinations of drug in which at least one of the drugs doesn't have DrugBank ID
dcombs_to_remove = unique(ct_studies_interventions_dcombo[which(is.na(ct_studies_interventions_dcombo$drugbank_id)), intervention_rxnorm_code_combination])
ct_studies_interventions_dcombo = ct_studies_interventions_dcombo %>% 
  filter(!intervention_rxnorm_code_combination %in% dcombs_to_remove)
rm(dcombs_to_remove)

ct_studies_interventions_dcombo = ct_studies_interventions_dcombo %>%
  dplyr::select(-rxnorm_code_per_drug) %>%
  group_by(intervention_rxnorm_code_combination, mesh_code) %>%
  mutate(drug_num = row_number()) %>%
  ungroup() %>%
  tidyr::pivot_wider(names_from = drug_num,
                     values_from = drugbank_id,
                     names_prefix = "drug") %>%
  dplyr::select(drug1, drug2, drug3, drug4, drug5, mesh_code) %>%
  distinct()
ct_studies_interventions_dcombo = ct_studies_interventions_dcombo[rowSums(!is.na(ct_studies_interventions_dcombo)) == 3,]
rownames(ct_studies_interventions_dcombo) = NULL
ct_studies_interventions_dcombo = ct_studies_interventions_dcombo %>% 
  dplyr::select(drug1, drug2, condition = mesh_code) %>%
  distinct()

ct_studies_manual_dcombo = ct_studies_manual_dcombo %>%
  dplyr::select(drug1 = DB_drug1, drug2 = DB_drug2, condition = mesh_code) %>%
  distinct()

ct_studies_interventions_dcombo_ultimate = rbind(ct_studies_interventions_dcombo, ct_studies_manual_dcombo)
ct_studies_interventions_dcombo_ultimate = unique(ct_studies_interventions_dcombo_ultimate)

## save files
fwrite(ct_studies_interventions_ingredients, paste0(ct_dir, "drug_condition.txt"), sep="\t", row.names = FALSE)
fwrite(ct_studies_interventions_dcombo_ultimate, paste0(ct_dir, "dcombinations_w_conditions.txt"), sep="\t", row.names = FALSE)
