##### GET PARTICIPANTS WITH COMPLETE DATA ##########
##### Milena Musial ################################
##### 09 - 2024 ####################################

rm(list = ls(all = TRUE))

##### load packages

packages <- c("dplyr", "tidyr", "lme4", "forcats")
# install.packages(packages)
sapply(packages, require, character.only = TRUE)

##### define paths

#data_path  <- "~/work/group_folder/B01_FP2_WP3/WP3_DATA/FINAL_STUDY"
data_path <- "/Users/milenamusial/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/PhD/04_B01/WP3/WP3_DATA/FINAL_STUDY"

##### read IDs that should be included (approved on Prolific, will be used to filter RedCap dfs)

# alcohol version - lowrisk drinkers
prolific_alc_lowrisk_1 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SRepresent_8_alc_lowrisk.csv"))
prolific_alc_lowrisk_1 <- prolific_alc_lowrisk_1 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "low-risk",
         sample = "GER")

prolific_alc_lowrisk_2 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SRepresent_12_alc_lowrisk.csv"))
prolific_alc_lowrisk_2 <- prolific_alc_lowrisk_2 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "low-risk",
         sample = "GER")

prolific_alc_lowrisk_3 <- read.csv(file.path(data_path, "inclusion_data/Prolific_BarNavigator_alc_lowrisk.csv"))
prolific_alc_lowrisk_3 <- prolific_alc_lowrisk_3 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "low-risk",
         sample = "GER")

prolific_alc_lowrisk_4 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SaturdayNightOut1_alc_lowrisk.csv"))
prolific_alc_lowrisk_4 <- prolific_alc_lowrisk_4 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "low-risk",
         sample = "UK")

prolific_alc_lowrisk_5 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SaturdayNightOut1a_alc_lowrisk.csv"))
prolific_alc_lowrisk_5 <- prolific_alc_lowrisk_5 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "low-risk",
         sample = "UK")

prolific_alc_lowrisk_6 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SaturdayNightOut4_alc_lowrisk.csv"))
prolific_alc_lowrisk_6 <- prolific_alc_lowrisk_6 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "low-risk",
         sample = "UK")

# alcohol version - harmful drinkers
prolific_alc_harmful_1 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SRepresent_9_alc_highrisk.csv"))
prolific_alc_harmful_1 <- prolific_alc_harmful_1 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "harmful",
         sample = "GER")

prolific_alc_harmful_2 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SRepresent_13_alc_highrisk.csv"))
prolific_alc_harmful_2 <- prolific_alc_harmful_2 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "harmful",
         sample = "GER")

prolific_alc_harmful_3 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SaturdayNightOut2_alc_highrisk.csv"))
prolific_alc_harmful_3 <- prolific_alc_harmful_3 %>%
    filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "harmful",
         sample = "UK")

prolific_alc_harmful_4 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SaturdayNightOut2a_alc_highrisk.csv"))
prolific_alc_harmful_4 <- prolific_alc_harmful_4 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "harmful",
         sample = "UK")

prolific_alc_harmful_5 <- read.csv(file.path(data_path, "inclusion_data/Prolific_SaturdayNightOut3_alc_highrisk.csv"))
prolific_alc_harmful_5 <- prolific_alc_harmful_5 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "alcohol",
         group = "harmful",
         sample = "UK")

prolific_alc_lowrisk <- rbind(prolific_alc_lowrisk_1, prolific_alc_lowrisk_2, prolific_alc_lowrisk_3, prolific_alc_lowrisk_4, prolific_alc_lowrisk_5, prolific_alc_lowrisk_6)
prolific_alc_lowrisk <- prolific_alc_lowrisk %>%
  rename(age_prolific = Age,
         ethnicity = Ethnicity.simplified,
         country_birth = Country.of.birth,
         country_residence = Country.of.residence,
         nationality = Nationality,
         native_language = Language,
         student = Student.status,
         employment = Employment.status)

prolific_alc_harmful <- rbind(prolific_alc_harmful_1, prolific_alc_harmful_2, prolific_alc_harmful_3, prolific_alc_harmful_4, prolific_alc_harmful_5)
prolific_alc_harmful <- prolific_alc_harmful %>%
  filter(Participant.id != "65b7c91aff83464ab6fc07a7") %>% # approved but no valid data
  rename(age_prolific = Age,
         ethnicity = Ethnicity.simplified,
         country_birth = Country.of.birth,
         country_residence = Country.of.residence,
         nationality = Nationality,
         native_language = Language,
         student = Student.status,
         employment = Employment.status) %>%
  unique() # exclude dublication of 6658b161506a8a9c5f1e4b58 who took part twice accidentally

# insert IDs for TRR participants that did not participate via Prolific
prolific_alc_harmful <- prolific_alc_harmful %>%
  add_row(Participant.id = "TRRalc001", country_residence = "Germany", version = "alcohol", group = "harmful", sample = "GER") %>%
  add_row(Participant.id = "TRRalc002", country_residence = "Germany", version = "alcohol", group = "harmful", sample = "GER") %>%
  add_row(Participant.id = "TRRalc003", country_residence = "Germany", version = "alcohol", group = "harmful", sample = "GER")

# control version - lowrisk drinkers
prolific_con_lowrisk_1 <- read.csv(file.path(data_path, "inclusion_data/Prolific_WeekendCashHunt1_control_lowrisk.csv"))
prolific_con_lowrisk_1 <- prolific_con_lowrisk_1 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "low-risk",
         sample = "GER")

prolific_con_lowrisk_2 <- read.csv(file.path(data_path, "inclusion_data/Prolific_WeekendCashHunt1a_control_lowrisk.csv"))
prolific_con_lowrisk_2 <- prolific_con_lowrisk_2 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "low-risk",
         sample = "EU")

prolific_con_lowrisk_3 <- read.csv(file.path(data_path, "inclusion_data/Prolific_WeekendCashHunt3_control_lowrisk.csv"))
prolific_con_lowrisk_3 <- prolific_con_lowrisk_3 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "low-risk",
         sample = "GER")

prolific_con_lowrisk_4 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch1_control_lowrisk.csv"))
prolific_con_lowrisk_4 <- prolific_con_lowrisk_4 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "low-risk",
         sample = "UK")

prolific_con_lowrisk_5 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch1a_control_lowrisk.csv"))
prolific_con_lowrisk_5 <- prolific_con_lowrisk_5 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "low-risk",
         sample = "UK")

prolific_con_lowrisk_6 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch4_control_lowrisk.csv"))
prolific_con_lowrisk_6 <- prolific_con_lowrisk_6 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "low-risk",
         sample = "UK")

# control version - harmful drinkers 
prolific_con_harmful_1 <- read.csv(file.path(data_path, "inclusion_data/Prolific_WeekendCashHunt2_control_highrisk.csv"))
prolific_con_harmful_1 <- prolific_con_harmful_1 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "GER")

prolific_con_harmful_2 <- read.csv(file.path(data_path, "inclusion_data/Prolific_WeekendCashHunt2a_control_highrisk.csv"))
prolific_con_harmful_2 <- prolific_con_harmful_2 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "EU")

prolific_con_harmful_3 <- read.csv(file.path(data_path, "inclusion_data/Prolific_WeekendCashHunt4_control_highrisk.csv"))
prolific_con_harmful_3 <- prolific_con_harmful_3 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "GER")

prolific_con_harmful_4 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch2_control_highrisk.csv"))
prolific_con_harmful_4 <- prolific_con_harmful_4 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "UK")

prolific_con_harmful_5 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch2a_control_highrisk.csv"))
prolific_con_harmful_5 <- prolific_con_harmful_5 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "UK")

prolific_con_harmful_6 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch2b_control_highrisk.csv"))
prolific_con_harmful_6 <- prolific_con_harmful_6 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "UK")

prolific_con_harmful_7 <- read.csv(file.path(data_path, "inclusion_data/Prolific_Cashsearch3_control_highrisk.csv"))
prolific_con_harmful_7 <- prolific_con_harmful_7 %>%
  filter(Status == "APPROVED") %>%
  select(Participant.id, Age, Ethnicity.simplified, Country.of.birth, Country.of.residence, Nationality, Language,
         Student.status, Employment.status) %>%
  mutate(version = "control",
         group = "harmful",
         sample = "UK")

prolific_con_lowrisk <- rbind(prolific_con_lowrisk_1, prolific_con_lowrisk_2, prolific_con_lowrisk_3, prolific_con_lowrisk_4, prolific_con_lowrisk_5, prolific_con_lowrisk_6)
prolific_con_lowrisk <- prolific_con_lowrisk %>%
  rename(age_prolific = Age,
         ethnicity = Ethnicity.simplified,
         country_birth = Country.of.birth,
         country_residence = Country.of.residence,
         nationality = Nationality,
         native_language = Language,
         student = Student.status,
         employment = Employment.status)

prolific_con_harmful <- rbind(prolific_con_harmful_1, prolific_con_harmful_2, prolific_con_harmful_3, prolific_con_harmful_4, prolific_con_harmful_5, prolific_con_harmful_6, prolific_con_harmful_7)
prolific_con_harmful <- prolific_con_harmful %>%
  rename(age_prolific = Age,
         ethnicity = Ethnicity.simplified,
         country_birth = Country.of.birth,
         country_residence = Country.of.residence,
         nationality = Nationality,
         native_language = Language,
         student = Student.status,
         employment = Employment.status)

# insert IDs for TRR participants that did not participate via Prolific
prolific_con_harmful <- prolific_con_harmful %>%
  add_row(Participant.id = "TRRcon001", country_residence = "Germany", version = "control", group = "harmful", sample = "GER") %>%
  add_row(Participant.id = "TRRcon002", country_residence = "Germany", version = "control", group = "harmful", sample = "GER") %>%
  add_row(Participant.id = "TRRcon003", country_residence = "Germany", version = "control", group = "harmful", sample = "GER") %>%
  add_row(Participant.id = "TRRcon004", country_residence = "Germany", version = "control", group = "harmful", sample = "GER")

# exclude participants who were approved, but did not complete task and questionnaires
prolific_con_harmful <- prolific_con_harmful %>%
  filter(! Participant.id %in% c("667a7551a3666990fac80091"))
prolific_alc_harmful <- prolific_alc_harmful %>%
  filter(! Participant.id %in% c("678f950f15b33743ed9c40d7",
                                 "676b158076c67ea864bd43a5",
                                 "677c11bc346d7ba49dae4d55"))

# save output
save(prolific_alc_lowrisk, prolific_alc_harmful, prolific_con_lowrisk, prolific_con_harmful, file = file.path(data_path, "RDFs/IDs_complete.RData"))
