##### PREPARE DATA FRAMES FOR BEHAVIORAL ANALYSES ##
##### Milena Musial ################################
##### 09 - 2024 ####################################

###### Preparations ######

rm(list = ls(all = TRUE))

# load packages

packages <- c("dplyr", "ggplot2", "rjson", "ndjson", "jsonlite", "tidyr", "lme4", "forcats")
#install.packages(packages)
sapply(packages, require, character.only = TRUE)

# define paths

#data_path <- "~/work/group_folder/B01_FP2_WP3/WP3_DATA/FINAL_STUDY"
data_path <- "/Users/milenamusial/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/PhD/04_B01/WP3/WP3_DATA/FINAL_STUDY"

# read IDs that should be included (approved on Prolific, will be used to filter RedCap dfs)

load(file.path(data_path, "RDFs/IDs_complete.RData"))

# read in psychometric data

load(file.path(data_path, "RDFs/demo_psych_data_w_oldrating.RData"))

###### read in behavioral data from .json and convert to df ######

### DE alcohol version ###

# full alcohol df
data_raw_alcohol_DE <- readLines(file.path(data_path, "behavioral_data/jatos_results_data_alcohol_DE.txt"))
data_list_alcohol_DE <- lapply(data_raw_alcohol_DE,fromJSON)
data_list_of_df_alcohol_DE <- lapply(data_list_alcohol_DE, data.frame, stringsAsFactors = FALSE)
data_df_alcohol_DE <- bind_rows(data_list_of_df_alcohol_DE)
data_df_alcohol_DE <- data_df_alcohol_DE %>% 
  unnest(c(aggregate_results.trialResults, 
           tutorial_results.trialResults,
           rating_results.ratingResults),
         names_sep = "_",
         keep_empty = TRUE)
data_df_alcohol_DE <- data_df_alcohol_DE %>% 
  group_by(participant_ID, running_ID, component, aggregate_results.trial) %>%
  mutate(state_index = row_number()) %>%
  mutate(state_index = if_else(is.na(aggregate_results.trial), NA, state_index))

# filter included participants based on redcap df
data_df_alcohol_DE <- data_df_alcohol_DE %>%
  filter(participant_ID %in% demo_psych[demo_psych$version=="alcohol",]$participant_ID & running_ID %in% demo_psych[demo_psych$version=="alcohol",]$running_ID)

# read data from participants with old rating version
data_raw_alcohol_DE_oldrating <- readLines(file.path(data_path, "behavioral_data/old_alcohol_version/alcohol_jatos_results_data_20240822172007.txt"))
data_list_alcohol_DE_oldrating <- lapply(data_raw_alcohol_DE_oldrating,fromJSON)
data_list_of_df_alcohol_DE_oldrating <- lapply(data_list_alcohol_DE_oldrating, data.frame, stringsAsFactors = FALSE)
data_df_alcohol_DE_oldrating <- bind_rows(data_list_of_df_alcohol_DE_oldrating)
data_df_alcohol_DE_oldrating <- data_df_alcohol_DE_oldrating %>% 
  unnest(c(aggregate_results.trialResults, 
           tutorial_results.trialResults,
           rating_results.ratingResults),
         names_sep = "_",
         keep_empty = TRUE)
data_df_alcohol_DE_oldrating <- data_df_alcohol_DE_oldrating %>% 
  group_by(participant_ID, running_ID, component, aggregate_results.trial) %>%
  mutate(state_index = row_number()) %>%
  mutate(state_index = if_else(is.na(aggregate_results.trial), NA, state_index))

# filter participants with old rating version based on redcap df
data_df_alcohol_DE_oldrating <- data_df_alcohol_DE_oldrating %>%
  filter(participant_ID %in% demo_psych[demo_psych$version=="alcohol",]$participant_ID & running_ID %in% demo_psych[demo_psych$version=="alcohol",]$running_ID)

# get IDs with old rating version
oldrating_IDs <- unique(data_df_alcohol_DE_oldrating$participant_ID)

# write info about rating into full alcohol df
data_df_alcohol_DE <- data_df_alcohol_DE %>%
  mutate(rating_version = ifelse(participant_ID %in% oldrating_IDs, "old", "new"),
         participant_ID = as.factor(participant_ID),
         running_ID = as.factor(running_ID))

# merge with prolific_id, group, version info
data_df_alcohol_DE <- inner_join(demo_psych[demo_psych$version=="alcohol",], data_df_alcohol_DE, by=c("participant_ID", "running_ID"))

### EN alcohol version ###

# full alcohol df
data_raw_alcohol_EN <- readLines(file.path(data_path, "behavioral_data/jatos_results_data_alcohol_EN.txt"))
data_list_alcohol_EN <- lapply(data_raw_alcohol_EN,fromJSON)
data_list_of_df_alcohol_EN <- lapply(data_list_alcohol_EN, data.frame, stringsAsFactors = FALSE)
data_df_alcohol_EN <- bind_rows(data_list_of_df_alcohol_EN)
data_df_alcohol_EN <- data_df_alcohol_EN %>% 
  unnest(c(aggregate_results.trialResults, 
           tutorial_results.trialResults,
           rating_results.ratingResults),
         names_sep = "_",
         keep_empty = TRUE)
data_df_alcohol_EN <- data_df_alcohol_EN %>%
  group_by(participant_ID, running_ID, component, aggregate_results.trial) %>%
  mutate(state_index = row_number()) %>%
  mutate(state_index = if_else(is.na(aggregate_results.trial), NA, state_index)) %>%
  mutate(rating_version = "new")

# filter included participants based on redcap df
data_df_alcohol_EN <- data_df_alcohol_EN %>%
  filter(participant_ID %in% demo_psych[demo_psych$version=="alcohol",]$participant_ID & running_ID %in% demo_psych[demo_psych$version=="alcohol",]$running_ID) %>%
  mutate(participant_ID = as.factor(participant_ID),
         running_ID = as.factor(running_ID))

# merge with prolific_id, group, version info
data_df_alcohol_EN <- inner_join(demo_psych[demo_psych$version=="alcohol",], data_df_alcohol_EN, by=c("participant_ID", "running_ID"))

### DE control version ###

# full control df
data_raw_control_DE <- readLines(file.path(data_path, "behavioral_data/jatos_results_data_control_DE.txt"))
data_list_control_DE <- lapply(data_raw_control_DE,fromJSON)
data_list_of_df_control_DE <- lapply(data_list_control_DE, data.frame, stringsAsFactors = FALSE)
data_df_control_DE <- bind_rows(data_list_of_df_control_DE)
data_df_control_DE <- data_df_control_DE %>% 
  unnest(c(aggregate_results.trialResults, 
           tutorial_results.trialResults,
           rating_results.ratingResults),
         names_sep = "_",
         keep_empty = TRUE)
data_df_control_DE <- data_df_control_DE %>%
  group_by(participant_ID, running_ID, component, aggregate_results.trial) %>%
  mutate(state_index = row_number()) %>%
  mutate(state_index = if_else(is.na(aggregate_results.trial), NA, state_index)) %>%
  mutate(drink = NA,
         rating_version = "new")

# filter included participants based on redcap df
data_df_control_DE <- data_df_control_DE %>%
  filter(participant_ID %in% demo_psych[demo_psych$version=="control",]$participant_ID & running_ID %in% demo_psych[demo_psych$version=="control",]$running_ID) %>%
  mutate(participant_ID = as.factor(participant_ID),
         running_ID = as.factor(running_ID))

# merge with prolific_id, group, version info
data_df_control_DE <- inner_join(demo_psych[demo_psych$version=="control",], data_df_control_DE, by=c("participant_ID", "running_ID"))

### EN control version ###

# full control df
data_raw_control_EN <- readLines(file.path(data_path, "behavioral_data/jatos_results_data_control_EN.txt"))
data_list_control_EN <- lapply(data_raw_control_EN,fromJSON)
data_list_of_df_control_EN <- lapply(data_list_control_EN, data.frame, stringsAsFactors = FALSE)
data_df_control_EN <- bind_rows(data_list_of_df_control_EN)
data_df_control_EN <- data_df_control_EN %>% 
  unnest(c(aggregate_results.trialResults, 
           tutorial_results.trialResults,
           rating_results.ratingResults),
         names_sep = "_",
         keep_empty = TRUE)
data_df_control_EN <- data_df_control_EN %>%
  group_by(participant_ID, running_ID, component, aggregate_results.trial) %>%
  mutate(state_index = row_number()) %>%
  mutate(state_index = if_else(is.na(aggregate_results.trial), NA, state_index)) %>%
  mutate(drink = NA,
         rating_version = "new")

# filter included participants based on redcap df
data_df_control_EN <- data_df_control_EN %>%
  filter(participant_ID %in% demo_psych[demo_psych$version=="control",]$participant_ID & running_ID %in% demo_psych[demo_psych$version=="control",]$running_ID) %>%
  mutate(participant_ID = as.factor(participant_ID),
         running_ID = as.factor(running_ID))

# merge with prolific_id, group, version info
data_df_control_EN <- inner_join(demo_psych[demo_psych$version=="control",], data_df_control_EN, by=c("participant_ID", "running_ID"))

# check that length of unique IDs equals prolific IDs with complete data
data_df_alcohol <- rbind(data_df_alcohol_DE, data_df_alcohol_EN)
data_df_control <- rbind(data_df_control_DE, data_df_control_EN)

length(unique(data_df_alcohol[data_df_alcohol$group=="harmful",]$prolific_ID)) == length(unique(prolific_alc_harmful$Participant.id))
length(unique(data_df_alcohol[data_df_alcohol$group=="low-risk",]$prolific_ID)) == length(unique(prolific_alc_lowrisk$Participant.id))
length(unique(data_df_control[data_df_control$group=="harmful",]$prolific_ID)) == length(unique(prolific_con_harmful$Participant.id))
length(unique(data_df_control[data_df_control$group=="low-risk",]$prolific_ID)) == length(unique(prolific_con_lowrisk$Participant.id))

##### Combine control and alcohol dfs
data_df <- rbind(data_df_control, data_df_alcohol)

# clean workspace
rm(list=setdiff(ls(), c("data_df", "demo_psych", "data_path")))

##############################  corrections #############################

# insert corrected variation ids

data_df$variation[data_df$prolific_ID == "634d7398662c6fe8321aa554"] <- "F3"
         
##############################  trial-df   ############################## 

# basic restructuring
trial_df <- data_df %>%
  select(prolific_ID,
         group,
         version,
         sample,
         rating_version,
         audit_sum_pre,
         audit_sum_post,
         aud_sum,
         aud_group,
         drinks_per_day,
         drinking_days,
         binge_days,
         uppsp_total,
         oci_total,
         casa_gf_unaware,
         casa_gf_nonvolitional,
         component,
         variation,
         correct_first_state_action,
         aggregate_results.trial,
         aggregate_results.trialResults_state,
         state_index,
         aggregate_results.trialResults_valid_choice,
         aggregate_results.trialResults_choice,
         aggregate_results.trialResults_RT,
         rating_version,
         rating_results.rating,
         rating_results.ratingResults_state,
         rating_results.ratingResults_value,
         rating_results.ratingResults_RT
         ) %>%
  rename(ID = prolific_ID,
         trial = aggregate_results.trial,
         state = aggregate_results.trialResults_state,
         valid_choice = aggregate_results.trialResults_valid_choice,
         choice = aggregate_results.trialResults_choice,
         RT = aggregate_results.trialResults_RT,
         rating_no = rating_results.rating,
         rating_state = rating_results.ratingResults_state,
         rating_value = rating_results.ratingResults_value,
         rating_RT = rating_results.ratingResults_RT
         )

trial_df <- trial_df %>%
  filter(! component %in% c("intro1", 
                            "intro2", 
                            "intro3",
                            "intro4",
                            "floor-plan",
                            "drink-selection",
                            "tutorial", 
                            "quiz", 
                            "quiz_wrong",
                            "outro",
                            "interlude-1",
                            "interlude-2",
                            "interlude-3",
                            "interlude-4")) %>%
  mutate_at(c('ID', 'component'), as.factor)

# create phase and condition variables
trial_df <- trial_df %>%
   mutate(phase = if_else(component %in% c("control-learning", 
                                          "reward-learning", 
                                          "transition-learning", 
                                          "goal-state-learning",
                                          "policy-learning"), "learning", 
                         if_else(component %in% c("control-relearning",
                                                  "reward-relearning",
                                                  "transition-relearning",
                                                  "goal-state-relearning",
                                                  "policy-relearning"), "relearning", 
                                 if_else(component %in% c("control-test",
                                                          "reward-test",
                                                          "transition-test",
                                                          "goal-state-test",
                                                          "policy-test"), "test", 
                                         if_else(component %in% c("control-rating",
                                                                  "reward-rating",
                                                                  "transition-rating",
                                                                  "goal-state-rating",
                                                                  "policy-rating"), "rating", 
                                                 "other")))),
         condition = if_else(component %in% c("control-learning",
                                              "control-relearning",
                                              "control-test",
                                              "control-rating"), "control", 
                             if_else(component %in% c("reward-learning",
                                                      "reward-relearning",
                                                      "reward-test",
                                                      "reward-rating"), "reward",
                                    if_else(component %in% c("transition-learning",
                                                              "transition-relearning",
                                                              "transition-test",
                                                              "transition-rating"), "transition", 
                                             if_else(component %in% c("goal-state-learning",
                                                                      "goal-state-relearning",
                                                                      "goal-state-test",
                                                                      "goal-state-rating"), "goal-state",
                                                     if_else(component %in% c("policy-learning",
                                                                              "policy-relearning",
                                                                              "policy-test",
                                                                              "policy-rating"), "policy", 
                                                             "other"))))))

# create condition index variable
trial_df <- trial_df %>%
   mutate(condition_index = case_when(((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                         "F1", "F2", "F3", "F4",
                                                         "G1", "G2", "G3", "G4", "G5", "G6", 
                                                         "G34", "G35", "G18", "G19", "G20", "G21") & condition == "reward") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "transition") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "policy") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "goal-state") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "control")) ~ 1,
                                      
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "transition") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "policy") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "goal-state") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "control") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "reward")) ~ 2,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "policy") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "goal-state") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "control") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "reward") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "transition")) ~ 3,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "goal-state") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "control") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "reward") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "transition") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "policy")) ~ 4,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "control") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "reward") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "transition") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "policy") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "goal-state")) ~ 5)
         )

# create environment variable
trial_df <- trial_df %>%
                                  # control version
  
  mutate(environment = case_when(((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "G2", "G35", "G7", "G8", "G12", "G17", "G41", "G23", "G26") & condition_index == 1) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G9", "G10", "G11", "G40", "G25", "G29", "G30") & condition_index == 5) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "G5", "G6", "G34", "G37", "G38", "G39", "G13", "G16", "G18", "G19", "G24", "G32") & condition_index == 4) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "G3", "G4", "G36", "G15", "G20", "G27", "G31", "G33") & condition_index == 3) |
                                    (version == "control" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "G1", "G14", "G21", "G22", "G28") & condition_index == 2)) ~ "white modern",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "G37", "G7", "G38", "G8", "G10", "G12", "G39", "G17", "G40", "G23", "G24", "G26") & condition_index == 2) |
                                    (version == "control" & variation %in% c("B1", "B2", "B3", "B4", "B5",
                                                                             "G9", "G11", "G14", "G16", "G30", "G31") & condition_index == 1) |
                                    (version == "control" & variation %in% c("C1", "D2", "D3", "D4", "D5", "E2", "E3",
                                                                             "G1", "G2", "G3", "G4", "G5", "G6", "G35", "G36", "G13", "G15", "G41", 
                                                                             "G18", "G19", "G27", "G28", "G32") & condition_index == 5) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "D1", "E4",
                                                                             "G20", "G21", "G22", "G29", "G33") & condition_index == 4) |
                                    (version == "control" & variation %in% c("E1", "E5", 
                                                                             "G34", "G25") & condition_index == 3)) ~ "blue floral",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3", "E4",
                                                                           "G5", "G6", "G35", "G11", "G12", "G39", "G17", "G40", "G22", "G24", "G11", "G32") & condition_index == 3) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G34", "G9", "G15", "G25", "G29") & condition_index == 2) |
                                    (version == "control" & variation %in% c("C1", "C2", "C3", "C4", "C5",
                                                                             "G36", "G38", "G10", "G13", "G18", "G19", "G21", "G28", "G33") & condition_index == 1) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "D1",
                                                                             "G37", "G8", "G16", "G20", "G26", "G31", "") & condition_index == 5) |
                                    (version == "control" & variation %in% c("D2", "D3", "D4", "D5", "E1", "E5",
                                                                             "G1", "G2", "G3", "G4", "G7", "G14", "G41", "G23", "G27", "G30") & condition_index == 4)) ~ "messy green",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3",
                                                                           "G35", "G36", "G9", "G11", "G12", "G17", "G40", "G25", "G26", "G28") & condition_index == 4) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G8", "G10", "G16", "G41", "G18", "G23", "G30") & condition_index == 3) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "G2", "G4", "G5", "G13", "G19", "G20", "G31") & condition_index == 2) |
                                    (version == "control" & variation %in% c("D1", "D2", "D3", "D4", "D5",
                                                                             "G1", "G3", "G6", "G15", "G34", "G37", "G39", "G27", "G29", "G32") & condition_index == 1) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "E1", "E4", "E5",
                                                                             "G7", "G38", "G14", "G21", "G22", "G24", "G33") & condition_index == 5)) ~ "orange tile",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "G34", "G12", "G39", "G17", "G23") & condition_index == 5) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G8", "G10", "G15", "G31") & condition_index == 4) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "G1", "G2", "G37", "G7", "G38", "G9", "G13", "G14", "G19", "G21", "G26", "G28", "G29") & condition_index == 3) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "G3", "G6", "G35", "G36", "G11", "G16", "G41", "G18", "G11", "G27", "G30", "G32", "G33") & condition_index == 2) |
                                    (version == "control" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "G4", "G5", "G40", "G20", "G22", "G24", "G25") & condition_index == 1)) ~ "red brown",
                                 
                                 # alcohol version
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "F4", "F7") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F6", "F8", "F14", "F15", "F19", "F21") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "F10", "F12", "F16") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "F1", "F2", "F3", "F9", "F11", "F13", "F17") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "F5", "F18", "F20") & condition_index == 2)) ~ "alternative",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "F7", "F10", "F12", "F14", "F21") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("B1", "B2", "B3", "B4", "B5",
                                                                             "F8", "F13", "F16") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("C1", "D2", "D3", "D4", "D5", "E2", "E3",
                                                                             "F11") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "D1", "E4",
                                                                             "F1", "F2", "F3", "F6", "F9", "F15", "F17") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("E1", "E5",
                                                                             "F4", "F5", "F18", "F19", "F20") & condition_index == 3)) ~ "brauhaus",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3", "E4",
                                                                           "F7") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F3", "F6", "F8", "F19") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("C1", "C2", "C3", "C4", "C5",
                                                                             "F10", "F15", "F17", "F18", "F20") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "D1",
                                                                             "F1", "F2", "F9", "F12", "F16") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("D2", "D3", "D4", "D5", "E1", "E5",
                                                                             "F4", "F5", "F11", "F13", "F14", "F21") & condition_index == 4)) ~ "fancy green",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3",
                                                                           "F7", "F8", "F19", "F20") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F6") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "F1", "F15", "F16") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("D1", "D2", "D3", "D4", "D5",
                                                                             "F2", "F9", "F11", "F12", "F14", "F21") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "E1", "E4", "E5",
                                                                             "F3", "F4", "F5", "F10", "F13", "F17", "F18") & condition_index == 5)) ~ "hip purple",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "F7", "F20") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F18") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "F8", "F10", "F12", "F14", "F15", "F16", "F21") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "F2", "F4", "F9", "F11", "F13", "F17") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "F1", "F3", "F5", "F6", "F19") & condition_index == 1)) ~ "sports bar"))

# ordering
trial_df <- trial_df %>%
  mutate(phase = fct_relevel(phase,
                                 c("learning",
                                   "relearning",
                                   "test",
                                   "rating"))) %>%
  arrange(ID, 
          condition_index,
          phase,
          trial,
          state_index)

# copy correct test-stage action from relearning to test and rating phases
for (n in unique(trial_df$ID)) {
  for (condition in c("control", "reward", "transition", "goal-state", "policy")) {
    trial_df$correct_first_state_action[trial_df$ID == n & trial_df$component == (paste(condition, "-test", sep=""))] = 
      trial_df$correct_first_state_action[trial_df$ID == n & trial_df$component == paste(condition, "-relearning", sep="")][1]
    trial_df$correct_first_state_action[trial_df$ID == n & trial_df$component == (paste(condition, "-rating", sep=""))] = 
      trial_df$correct_first_state_action[trial_df$ID == n & trial_df$component == paste(condition, "-relearning", sep="")][1]
  }
}

# calculate correct for every 2-choice state
trial_df <- trial_df %>%
  mutate(correct_second_state_action = case_when( 
                                                 # learning phase
                                                 (phase == "learning" & correct_first_state_action == "right" & condition %in% c("reward", "goal-state", "control")) ~ "left",
                                                 (phase == "learning" & correct_first_state_action == "right" & condition %in% c("transition", "policy")) ~ "right",
                                                 (phase == "learning" & correct_first_state_action == "left" & condition %in% c("reward", "goal-state", "control")) ~ "left",
                                                 (phase == "learning" & correct_first_state_action == "left" & condition %in% c("transition", "policy")) ~ "left",
                                                 
                                                 # test and rating phases
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "left" & condition %in% c("reward", "goal-state", "control")) ~ "left", # in control condition, left is always right from state 2 no matter what correct 1st state action is
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "left" & condition %in% c("transition", "policy")) ~ "left",
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "right" & condition %in% c("reward", "goal-state", "control")) ~ "left",
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "right" & condition %in% c("transition", "policy")) ~ "left"),
         correct_third_state_action = case_when(
                                                 # learning phase
                                                 (phase == "learning" & correct_first_state_action == "right" & condition %in% c("reward", "goal-state", "control")) ~ "right",
                                                 (phase == "learning" & correct_first_state_action == "right" & condition %in% c("transition", "policy")) ~ "right",
                                                 (phase == "learning" & correct_first_state_action == "left" & condition %in% c("reward", "goal-state", "control")) ~ "right",
                                                 (phase == "learning" & correct_first_state_action == "left" & condition %in% c("transition", "policy")) ~ "left",
                                                 
                                                 # test and rating phases
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "left" & condition %in% c("reward", "goal-state", "control")) ~ "right",
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "left" & condition %in% c("transition", "policy")) ~ "right",
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "right" & condition %in% c("reward", "goal-state", "control")) ~ "right",
                                                 (phase %in% c("test", "rating") & correct_first_state_action == "right" & condition %in% c("transition", "policy")) ~ "right")
         ) %>%
  mutate(correct_state_1 = if_else((state == 1 & correct_first_state_action == choice), 1,
                                   if_else((state == 1 & correct_first_state_action != choice), 0, NA)),
         correct_state_2 = if_else((state == 2 & correct_second_state_action == choice), 1,
                                   if_else((state == 2 & correct_second_state_action != choice), 0, NA)),
         correct_state_3 = if_else((state == 3 & correct_third_state_action == choice), 1,
                                   if_else((state == 3 & correct_third_state_action != choice), 0, NA))) %>%
  mutate(correct = coalesce(correct_state_1, correct_state_2, correct_state_3)) %>%
  # calculate switch for state 1 in test phase
  mutate(switch = if_else((state == 1 & component %in% c("control-test")), abs(correct-1), # in control condition, it is correct to not switch, so all incorrect trials are switch trials
                          if_else((state == 1 & phase %in% c("test")), correct, NA))) # in all other conditions, it is correct to switch, so all correct trials are switch trials

# add drink selection per participant
drink_df <- data_df %>%
  dplyr::select(prolific_ID, version, component, drink) %>%
  mutate(ID = prolific_ID) %>%
  filter(component == "drink-selection") %>%
  group_by(ID) %>%
  slice_tail(n = 1) %>% # 55d51a6b8ce09000127d4821 did drink selection twice
  select(ID, drink)

trial_df <- left_join(trial_df, drink_df, by = "ID")

# rename & reorder 
trial_df <- trial_df %>%
  mutate(version = case_when(
    version == "alcohol" ~ "Alcohol version",
    version == "control" ~ "Monetary version"
  ),
  group = case_when(
    group == "harmful" ~ "Harmful drinkers",
    group == "low-risk" ~ "Low-risk drinkers"
  ),
  condition = case_when(
    condition == "reward" ~ "Reward revaluation",
    condition == "transition" ~ "Transition revaluation",
    condition == "goal-state" ~ "Goal-state revaluation",
    condition == "policy" ~ "Policy revaluation",
    condition == "control" ~ "Control"
  )) %>%
  mutate(version = fct_relevel(version,
                                c("Monetary version",
                                  "Alcohol version")),
         group = fct_relevel(group,
                             c("Low-risk drinkers",
                               "Harmful drinkers")),
         condition = fct_relevel(condition,
                                 c("Reward revaluation",
                                   "Goal-state revaluation",
                                   "Transition revaluation",
                                   "Policy revaluation",
                                   "Control")))

# exclude trials with invalid choice
trial_df <- trial_df %>% # create running index
  mutate(running_index = c(1:nrow(trial_df)))

invalid_trial_df <- trial_df %>% # create df containing trials with invalid choice
  filter(valid_choice == FALSE) %>%
  arrange(ID, 
          condition_index,
          phase,
          trial,
          state_index)

for (i in invalid_trial_df$running_index) { # for every invalid trial identified by its running index
  
  # if state == 2 or 3 & trial same as in previous line, code valid_choice in previous line as FALSE
  if (trial_df$state[trial_df$running_index == i] %in% c(2,3,"2Left","2Right","3Left","3Right") & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i-1])) {
    trial_df$valid_choice[trial_df$running_index == i-1] = FALSE
  }
  
  # if state == 4, 5, or 6 & trial same as in second previous line, code valid_choice in previous and second previous line as FALSE
  if (trial_df$state[trial_df$running_index == i] %in% c(4,5,6) & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i-2])) {
    trial_df$valid_choice[trial_df$running_index == i-1] = FALSE
    trial_df$valid_choice[trial_df$running_index == i-2] = FALSE
  }
  
  # if state == 7, 8, or 9 & trial same as in third previous line, code valid_choice in previous, second previous, and third previous line as FALSE
  if (trial_df$state[trial_df$running_index == i] %in% c(7,8,9) & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i-3])) {
    trial_df$valid_choice[trial_df$running_index == i-1] = FALSE
    trial_df$valid_choice[trial_df$running_index == i-2] = FALSE
    trial_df$valid_choice[trial_df$running_index == i-3] = FALSE
  }
  
  # if state == 10 & trial same as in fourth previous line, code valid_choice in previous, second, third, and fourth previous line as FALSE
  if (trial_df$state[trial_df$running_index == i] == 10 & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i-4])) {
    trial_df$valid_choice[trial_df$running_index == i-1] = FALSE
    trial_df$valid_choice[trial_df$running_index == i-2] = FALSE
    trial_df$valid_choice[trial_df$running_index == i-3] = FALSE
    trial_df$valid_choice[trial_df$running_index == i-4] = FALSE
  }

  # relearning trials: start at state 4/5/6 (3 states per trial), not state 1
  if (trial_df$phase[trial_df$running_index == i] == "relearning") {

    if (trial_df$state[trial_df$running_index == i] %in% c(7, 8, 9) & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i - 1])) {
      trial_df$valid_choice[trial_df$running_index == i - 1] = FALSE
    }
    if (trial_df$state[trial_df$running_index == i] == 10 & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i - 1])) {
      trial_df$valid_choice[trial_df$running_index == i - 1] = FALSE
    }
    if (trial_df$state[trial_df$running_index == i] == 10 & (trial_df$trial[trial_df$running_index == i] %in% trial_df$trial[trial_df$running_index == i - 2])) {
      trial_df$valid_choice[trial_df$running_index == i - 2] = FALSE
    }
  }
}

# extract rows relevant for rating df (prepared below)
rating_df <- trial_df %>%
  filter(phase == "rating")

# exclude invalid trials as computed above and rating trials (valid choice always NA, as no trial number), compute accumulated trial per participant
trial_df <- trial_df %>%
  filter(valid_choice == TRUE) %>%
  arrange(ID, 
          condition_index,
          phase,
          trial,
          state_index)
  # group_by(ID) %>%
  # mutate(accumulated_states_visited = row_number())

# calculate if correct path taken from state 1
trial_df <- trial_df %>%
  mutate(correct_path = case_when((state == 1 & correct == 1 & lead(correct) == 1) ~ 1, # correct state 1 and second stage choice
                                  (state == 1 & correct == 0) ~ 0, # incorrect state 1 choice
                                  (state == 1 & lead(correct) == 0) ~ 0)) # incorrect second stage choice

# calculate if change in path preference occurred (for reward, goal-state, transition, policy == correct path; for control != correct path)
trial_df <- trial_df %>%
  mutate(switch_path = if_else((state == 1 & component %in% c("control-test")), abs(correct_path-1), # in control condition, it is correct to not switch, so all incorrect trials are switch trials
                        if_else((state == 1 & phase %in% c("test")), correct_path, NA))) # in all other conditions, it is correct to switch, so all correct trials are switch trials


# calculate reward received (we pretend reward is always monetary for both versions, as we assume that value of 1 glass == 15 euros etc.)
trial_df <- trial_df %>%
  mutate(
    reward = case_when(
      
      # learning reward condition
      condition %in% c("Reward revaluation", "Goal-state revaluation", "Control") & phase == "learning" & correct_first_state_action == "right" & state == "7" ~ 15,
      condition %in% c("Reward revaluation", "Goal-state revaluation", "Control") & phase == "learning" & correct_first_state_action == "right" & state == "9" ~ 30,
      condition %in% c("Reward revaluation", "Goal-state revaluation", "Control") & phase == "learning" & correct_first_state_action == "left"  & state == "7" ~ 30,
      condition %in% c("Reward revaluation", "Goal-state revaluation", "Control") & phase == "learning" & correct_first_state_action == "left"  & state == "9" ~ 15,
      
      condition %in% c("Transition revaluation", "Policy revaluation") & phase == "learning" & correct_first_state_action == "right" & state == "8" ~ 15,
      condition %in% c("Transition revaluation", "Policy revaluation") & phase == "learning" & correct_first_state_action == "right" & state == "9" ~ 30,
      condition %in% c("Transition revaluation", "Policy revaluation") & phase == "learning" & correct_first_state_action == "left"  & state == "7" ~ 30,
      condition %in% c("Transition revaluation", "Policy revaluation") & phase == "learning" & correct_first_state_action == "left"  & state == "8" ~ 15,
      
      # relearning reward
      condition == "Reward revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "7" ~ 45,
      condition == "Reward revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "9" ~ 30,
      condition == "Reward revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "7" ~ 30,
      condition == "Reward revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "9" ~ 45,
      
      # relearning goal-state
      condition == "Goal-state revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "4" ~ 45,
      condition == "Goal-state revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "7" ~ 15,
      condition == "Goal-state revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "9" ~ 30,
      condition == "Goal-state revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "6" ~ 45,
      condition == "Goal-state revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "7" ~ 30,
      condition == "Goal-state revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "9" ~ 15,
      
      # relearning transition
      condition == "Transition revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "8" ~ 15,
      condition == "Transition revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "9" ~ 30,
      condition == "Transition revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "7" ~ 30,
      condition == "Transition revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "8" ~ 15,
      
      # relearning policy
      condition == "Policy revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "7" ~ 45,
      condition == "Policy revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "8" ~ 15,
      condition == "Policy revaluation" & phase == "relearning" & correct_first_state_action == "left"  & state == "9" ~ 30,
      condition == "Policy revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "7" ~ 30,
      condition == "Policy revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "8" ~ 15,
      condition == "Policy revaluation" & phase == "relearning" & correct_first_state_action == "right" & state == "9" ~ 45,
      
      # relearning control
      condition == "Control" & phase == "relearning" & correct_first_state_action == "right" & state == "7" ~ 15,
      condition == "Control" & phase == "relearning" & correct_first_state_action == "right" & state == "9" ~ 45,
      condition == "Control" & phase == "relearning" & correct_first_state_action == "left"  & state == "7" ~ 45,
      condition == "Control" & phase == "relearning" & correct_first_state_action == "left"  & state == "9" ~ 15,
      
      TRUE ~ 0
    )
  )



# order columns
trial_df <- trial_df %>%
   select(ID, 
          group,
          version,
          sample,
          rating_version,
          audit_sum_pre,
          audit_sum_post,
          aud_sum,
          aud_group,
          drinks_per_day,
          drinking_days,
          binge_days,
          uppsp_total,
          oci_total,
          casa_gf_unaware,
          casa_gf_nonvolitional, 
          variation, drink, component, phase, condition, condition_index, environment,
          correct_first_state_action, correct_second_state_action, correct_third_state_action,
          trial, state, state_index, choice, valid_choice, RT, correct_state_1, correct_state_2, correct_state_3, 
          correct, reward, correct_path, switch, switch_path) %>%
    arrange(ID, 
          condition_index,
          phase,
          trial,
          state_index)

##############################  rating df ##############################  
        
rating_df <- rating_df %>%
  select(ID,
         group,
         version,
         sample,
         audit_sum_pre,
         audit_sum_post,
         aud_sum,
         aud_group,
         drinks_per_day,
         drinking_days,
         binge_days,
         uppsp_total,
         oci_total,
         casa_gf_unaware,
         casa_gf_nonvolitional, 
         variation,
         component,
         phase,
         condition,
         condition_index,
         environment,
         correct_first_state_action,
         correct_second_state_action,
         correct_third_state_action,
         rating_version,
         rating_no,
         rating_state,
         rating_value,
         rating_RT) %>%
  mutate(rating_value = as.numeric(rating_value),
         state = case_when((rating_state %in% c("1LeftRating", "1RightRating")) ~ 1,
                           (rating_state %in% c("2LeftRating", "2RightRating")) ~ 2,
                           (rating_state %in% c("3LeftRating", "3RightRating")) ~ 3)) %>%
  arrange(ID, component, rating_state) %>%
  group_by(ID, condition) %>%
  # compute difference between better and worse action per state
  # value in each row indicates how much better (positive) or worse (negative) the corresponding optimal option was rated compared to the suboptimal option
  mutate(rating_diff_state1 = case_when((rating_state == "1LeftRating" & correct_first_state_action == "left") ~ (rating_value - lead(rating_value)),
                                        (rating_state == "1RightRating" & correct_first_state_action == "right") ~ (rating_value - lag(rating_value))))
  
rating_df <- rating_df %>%
  mutate(rating_diff_switch_state1 = if_else((component %in% c("control-rating")), rating_diff_state1*(-1), # in control condition, it is correct to not switch, so all incorrect trials are switch trials
                               if_else((phase %in% c("rating")), rating_diff_state1, NA))) # in all other conditions, it is correct to switch, so all correct trials are switch trials



##############################  config-df with info on randomization   ##############################  
component_df <- data_df %>%
  select(prolific_ID,
         back_code,
         group,
         version,
         sample,
         rating_version,
         audit_sum_pre,
         audit_sum_post,
         aud_sum,
         aud_group,
         drinks_per_day,
         drinking_days,
         binge_days,
         uppsp_total,
         oci_total,
         casa_gf_unaware,
         casa_gf_nonvolitional, 
         variation,
         component,
         component_duration,
         state_room_map.1,
         state_room_map.2,
         state_room_map.3,
         state_room_map.4,
         state_room_map.5,
         state_room_map.6,
         state_room_map.7,
         state_room_map.8,
         state_room_map.9,
         state_room_map.10) %>%
  rename(ID = prolific_ID) %>%
  distinct() %>%
  arrange(ID,
          component) %>%
  mutate_at(c('ID', 'component'), as.factor) %>%
  mutate(phase = if_else(component %in% c("control-learning", 
                                          "reward-learning", 
                                          "transition-learning", 
                                          "goal-state-learning",
                                          "policy-learning"), "learning", 
                         if_else(component %in% c("control-relearning",
                                                  "reward-relearning",
                                                  "transition-relearning",
                                                  "goal-state-relearning",
                                                  "policy-relearning"), "relearning", 
                                 if_else(component %in% c("control-test",
                                                          "reward-test",
                                                          "transition-test",
                                                          "goal-state-test",
                                                          "policy-test"), "test", 
                                         if_else(component %in% c("control-rating",
                                                                  "reward-rating",
                                                                  "transition-rating",
                                                                  "goal-state-rating",
                                                                  "policy-rating"), "rating", 
                                                 "other")))),
         condition = if_else(component %in% c("control-learning",
                                              "control-relearning",
                                              "control-test",
                                              "control-rating"), "control", 
                             if_else(component %in% c("reward-learning",
                                                      "reward-relearning",
                                                      "reward-test",
                                                      "reward-rating"), "reward",
                                     if_else(component %in% c("transition-learning",
                                                              "transition-relearning",
                                                              "transition-test",
                                                              "transition-rating"), "transition", 
                                             if_else(component %in% c("goal-state-learning",
                                                                      "goal-state-relearning",
                                                                      "goal-state-test",
                                                                      "goal-state-rating"), "goal-state",
                                                     if_else(component %in% c("policy-learning",
                                                                              "policy-relearning",
                                                                              "policy-test",
                                                                              "policy-rating"), "policy", 
                                                             "other")))))) %>%
  mutate(condition_index = case_when(((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "reward") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "transition") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "policy") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "goal-state") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "control")) ~ 1,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "transition") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "policy") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "goal-state") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "control") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "reward")) ~ 2,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "policy") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "goal-state") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "control") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "reward") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "transition")) ~ 3,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "goal-state") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "control") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "reward") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "transition") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "policy")) ~ 4,
                                     
                                     ((variation %in% c("A1", "B1", "C1", "D1", "E1", 
                                                        "F1", "F2", "F3", "F4",
                                                        "G1", "G2", "G3", "G4", "G5", "G6", 
                                                        "G34", "G35", "G18", "G19", "G20", "G21") & condition == "control") |
                                        (variation %in% c("A2", "B2", "C2", "D2", "E2",
                                                          "F5", "F17", "F18", "F19", 
                                                          "G36", "G37", "G22") & condition == "reward") |
                                        (variation %in% c("A3", "B3", "C3", "D3", "E3",
                                                          "F6", "F7", "F20",
                                                          "G7", "G38", "G23", "G24", "G25") & condition == "transition") |
                                        (variation %in% c("A4", "B4", "C4", "D4", "E4",
                                                          "F8", "F21",
                                                          "G8", "G9", "G10", "G11", "G12", "G39", "G26", "G27") & condition == "policy") |
                                        (variation %in% c("A5", "B5", "C5", "D5", "E5",
                                                          "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
                                                          "G13", "G14", "G15", "G16", "G17", "G40", "G41",
                                                          "G28", "G29", "G30", "G31", "G32", "G33") & condition == "goal-state")) ~ 5)
  ) %>%
  mutate(environment = case_when(((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "G2", "G35", "G7", "G8", "G12", "G17", "G41", "G23", "G26") & condition_index == 1) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G9", "G10", "G11", "G40", "G25", "G29", "G30") & condition_index == 5) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "G5", "G6", "G34", "G37", "G38", "G39", "G13", "G16", "G18", "G19", "G24", "G32") & condition_index == 4) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "G3", "G4", "G36", "G15", "G20", "G27", "G31", "G33") & condition_index == 3) |
                                    (version == "control" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "G1", "G14", "G21", "G22", "G28") & condition_index == 2)) ~ "white modern",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "G37", "G7", "G38", "G8", "G10", "G12", "G39", "G17", "G40", "G23", "G24", "G26") & condition_index == 2) |
                                    (version == "control" & variation %in% c("B1", "B2", "B3", "B4", "B5",
                                                                             "G9", "G11", "G14", "G16", "G30", "G31") & condition_index == 1) |
                                    (version == "control" & variation %in% c("C1", "D2", "D3", "D4", "D5", "E2", "E3",
                                                                             "G1", "G2", "G3", "G4", "G5", "G6", "G35", "G36", "G13", "G15", "G41", 
                                                                             "G18", "G19", "G27", "G28", "G32") & condition_index == 5) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "D1", "E4",
                                                                             "G20", "G21", "G22", "G29", "G33") & condition_index == 4) |
                                    (version == "control" & variation %in% c("E1", "E5", 
                                                                             "G34", "G25") & condition_index == 3)) ~ "blue floral",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3", "E4",
                                                                           "G5", "G6", "G35", "G11", "G12", "G39", "G17", "G40", "G22", "G24", "G11", "G32") & condition_index == 3) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G34", "G9", "G15", "G25", "G29") & condition_index == 2) |
                                    (version == "control" & variation %in% c("C1", "C2", "C3", "C4", "C5",
                                                                             "G36", "G38", "G10", "G13", "G18", "G19", "G21", "G28", "G33") & condition_index == 1) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "D1",
                                                                             "G37", "G8", "G16", "G20", "G26", "G31", "") & condition_index == 5) |
                                    (version == "control" & variation %in% c("D2", "D3", "D4", "D5", "E1", "E5",
                                                                             "G1", "G2", "G3", "G4", "G7", "G14", "G41", "G23", "G27", "G30") & condition_index == 4)) ~ "messy green",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3",
                                                                           "G35", "G36", "G9", "G11", "G12", "G17", "G40", "G25", "G26", "G28") & condition_index == 4) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G8", "G10", "G16", "G41", "G18", "G23", "G30") & condition_index == 3) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "G2", "G4", "G5", "G13", "G19", "G20", "G31") & condition_index == 2) |
                                    (version == "control" & variation %in% c("D1", "D2", "D3", "D4", "D5",
                                                                             "G1", "G3", "G6", "G15", "G34", "G37", "G39", "G27", "G29", "G32") & condition_index == 1) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "E1", "E4", "E5",
                                                                             "G7", "G38", "G14", "G21", "G22", "G24", "G33") & condition_index == 5)) ~ "orange tile",
                                 
                                 ((version == "control" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "G34", "G12", "G39", "G17", "G23") & condition_index == 5) |
                                    (version == "control" & variation %in% c("B1",
                                                                             "G8", "G10", "G15", "G31") & condition_index == 4) |
                                    (version == "control" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "G1", "G2", "G37", "G7", "G38", "G9", "G13", "G14", "G19", "G21", "G26", "G28", "G29") & condition_index == 3) |
                                    (version == "control" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "G3", "G6", "G35", "G36", "G11", "G16", "G41", "G18", "G11", "G27", "G30", "G32", "G33") & condition_index == 2) |
                                    (version == "control" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "G4", "G5", "G40", "G20", "G22", "G24", "G25") & condition_index == 1)) ~ "red brown",
                                 
                                 # alcohol version
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "F4", "F7") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F6", "F8", "F14", "F15", "F19", "F21") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "F10", "F12", "F16") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "F1", "F2", "F3", "F9", "F11", "F13", "F17") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "F5", "F18", "F20") & condition_index == 2)) ~ "alternative",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "F7", "F10", "F12", "F14", "F21") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("B1", "B2", "B3", "B4", "B5",
                                                                             "F8", "F13", "F16") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("C1", "D2", "D3", "D4", "D5", "E2", "E3",
                                                                             "F11") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "D1", "E4",
                                                                             "F1", "F2", "F3", "F6", "F9", "F15", "F17") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("E1", "E5",
                                                                             "F4", "F5", "F18", "F19", "F20") & condition_index == 3)) ~ "brauhaus",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3", "E4",
                                                                           "F7") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F3", "F6", "F8", "F19") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("C1", "C2", "C3", "C4", "C5",
                                                                             "F10", "F15", "F17", "F18", "F20") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "D1",
                                                                             "F1", "F2", "F9", "F12", "F16") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("D2", "D3", "D4", "D5", "E1", "E5",
                                                                             "F4", "F5", "F11", "F13", "F14", "F21") & condition_index == 4)) ~ "fancy green",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5", "E2", "E3",
                                                                           "F7", "F8", "F19", "F20") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F6") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "F1", "F15", "F16") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("D1", "D2", "D3", "D4", "D5",
                                                                             "F2", "F9", "F11", "F12", "F14", "F21") & condition_index == 1) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "E1", "E4", "E5",
                                                                             "F3", "F4", "F5", "F10", "F13", "F17", "F18") & condition_index == 5)) ~ "hip purple",
                                 
                                 ((version == "alcohol" & variation %in% c("A1", "A2", "A3", "A4", "A5",
                                                                           "F7", "F20") & condition_index == 5) |
                                    (version == "alcohol" & variation %in% c("B1",
                                                                             "F18") & condition_index == 4) |
                                    (version == "alcohol" & variation %in% c("B2", "B3", "B4", "B5", "C1",
                                                                             "F8", "F10", "F12", "F14", "F15", "F16", "F21") & condition_index == 3) |
                                    (version == "alcohol" & variation %in% c("C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
                                                                             "F2", "F4", "F9", "F11", "F13", "F17") & condition_index == 2) |
                                    (version == "alcohol" & variation %in% c("E1", "E2", "E3", "E4", "E5",
                                                                             "F1", "F3", "F5", "F6", "F19") & condition_index == 1)) ~ "sports bar"))

component_df <- merge(component_df, drink_df, by = "ID", all.x = T)

##############################  filter all dfs to contain only new rating version subjects   ##############################  

trial_df_w_oldrating <- trial_df
trial_df <- trial_df %>%
  filter(rating_version == "new")

rating_df_w_oldrating <- rating_df
rating_df <- rating_df %>%
  filter(rating_version == "new")

component_df_w_oldrating <- component_df
component_df <- component_df %>%
  filter(rating_version == "new")

demo_psych <- demo_psych %>%
  filter(prolific_ID %in% trial_df$ID)

##############################  save   ##############################  
save(trial_df, rating_df, component_df, file = file.path(data_path, "RDFs/final_data_complete.RData"))
save(trial_df_w_oldrating, rating_df_w_oldrating, component_df_w_oldrating, file = file.path(data_path, "RDFs/final_data_complete_w_oldrating.RData"))
save(demo_psych, file = file.path(data_path, "RDFs/demo_psych_data.RData"))

