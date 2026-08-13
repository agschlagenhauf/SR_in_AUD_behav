#
# main.py
#

import os
import numpy as np
import pandas as pd
from os.path import join
from simulate import *
from utilities import *

# set seed for reproducibility
np.random.seed(42)

#
# Set simulation parameters
#

SIMULATION_MODE = "ppc" # "fixed" for multiple simulations from same set of manually defined parameter values; "ppc" for posterior predictive checks; "recovery" for one single simulation from set of parameter values drawn from distribution

# Hybrid MF models (hybrid_mf_*) require mix_w from fitted parameters and are
# only simulated in PPC mode.               

MODELS = [#"sr", 
          #"mb", 
          #"mf",
          #"redsr_2",
          #"redsr_3",
          #"redsr_4",
          #"randsr_noupdate",
          #"randsr_wupdate",
          #"redsr_2_randsr_wupdate",
          #"redsr_3_randsr_wupdate",
          #"redsr_4_randsr_wupdate",
          #"mb_learnt",
          "hybrid_mf_redsr_4_randsr_wupdate",
          "hybrid_mf_randsr_noupdate",
          "hybrid_mf_randsr_wupdate",
          "hybrid_mf_mb_learnt"
          ]

CONDITIONS = ["control", "reward", "transition", "policy", "goal"] # "control", "reward", "transition", "policy", "goal"

if SIMULATION_MODE == "fixed":
    OUTPUT_DIR = "results_fixed"
elif SIMULATION_MODE == "ppc":
    OUTPUT_DIR = "results_ppc"
    INPUT_FILE = "/Users/milenamusial/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/PhD/04_B01/WP3/SR_in_AUD_behav/04_model_fit/results_real_data/model_fit_per_participant_lbfgsb.csv"

#
# Transition Log Headers
#
TRANSITION_LOG_HEADER_PREFIX = "simulation_number,model,condition,phase,trial,state,action,reward"

def get_transition_log_headers():
    """
    Creates the header lines for a .csv file for each model type.

    Returns:
        - dict: maps model name to its .csv header line 
    """

    transition_log_headers = {}

    # Action choices (left = 1, right = 2, forced = 1)
    state_action_choices = {
        1:  [ACTION_LEFT, ACTION_RIGHT],
        2:  [ACTION_LEFT, ACTION_RIGHT],
        3:  [ACTION_LEFT, ACTION_RIGHT],
        4:  [ACTION_FORCED],
        5:  [ACTION_FORCED],
        6:  [ACTION_FORCED],
        7:  [ACTION_FORCED],
        8:  [ACTION_FORCED],
        9:  [ACTION_FORCED],
        10: [ACTION_FORCED]
    }

    # e.g. S1A2
    state_strings = []
    state_action_strings = []
    for state in range(len(state_action_choices)):
        state_strings.append(f"S{state+1}")
    for state_number, choices in state_action_choices.items():
        for choice in choices:
            state_action_strings.append(f"S{state_number}A{choice}")

    value_strings = [f"V{item}" for item in state_action_strings]
    value_strings_joined = ",".join(value_strings)

    mf_value_strings = [f"MFV{item}" for item in state_action_strings]
    mf_value_strings_joined = ",".join(mf_value_strings)

    weight_strings = [f"W{item}" for item in state_action_strings]
    weight_strings_joined = ",".join(weight_strings)

    # e.g. S1A2-S3A1
    state_action_combination_strings = []
    for first_state_action_pair in state_action_strings:
        for second_state_action_pair in state_action_strings:
            state_action_combination_strings.append(f"{first_state_action_pair}-{second_state_action_pair}")

    occupancy_strings = [f"O{item}" for item in state_action_combination_strings]
    occupancy_strings_joined = ",".join(occupancy_strings)

    # e.g. S1A2-S3
    state_action_state_strings = []
    for state_action_pair in state_action_strings:
        for state in state_strings:
            state_action_state_strings.append(f"{state_action_pair}-{state}")

    transition_strings = [f"T{item}" for item in state_action_state_strings]
    transition_strings_joined = ",".join(transition_strings)

    transition_log_headers["sr"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    
    transition_log_headers["randsr_noupdate"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    transition_log_headers["randsr_wupdate"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    
    transition_log_headers["redsr_2"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    transition_log_headers["redsr_3"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    transition_log_headers["redsr_4"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    
    transition_log_headers["redsr_2_randsr_wupdate"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    transition_log_headers["redsr_3_randsr_wupdate"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"
    transition_log_headers["redsr_4_randsr_wupdate"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{occupancy_strings_joined}\n"

    transition_log_headers["hybrid_mf_redsr_4_randsr_wupdate"] = (
        f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},"
        f"{occupancy_strings_joined},{mf_value_strings_joined}\n"
    )
    transition_log_headers["hybrid_mf_randsr_noupdate"] = (
        f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},"
        f"{occupancy_strings_joined},{mf_value_strings_joined}\n"
    )
    transition_log_headers["hybrid_mf_randsr_wupdate"] = (
        f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},"
        f"{occupancy_strings_joined},{mf_value_strings_joined}\n"
    )
    
    transition_log_headers["mb"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{transition_strings_joined}\n"
    transition_log_headers["mb_learnt"] = f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},{transition_strings_joined}\n"
    transition_log_headers["hybrid_mf_mb_learnt"] = (
        f"{TRANSITION_LOG_HEADER_PREFIX},weight_delta,{value_strings_joined},{weight_strings_joined},"
        f"{transition_strings_joined},{mf_value_strings_joined}\n"
    )
    
    transition_log_headers["mf"] = f"{TRANSITION_LOG_HEADER_PREFIX},{value_strings_joined}\n"

    return transition_log_headers

### EXECUTION: Iterate through parameter values indicated above ###
if __name__ == "__main__":
    
    if SIMULATION_MODE == "fixed":

        # Set number of simulations per parameter combination and model
        NUM_SIMULATIONS = 1000
        
        # Set parameter values to iterate over
        ALPHA_TD = [0.1, 0.3, 0.5, 0.7, 0.9]
        ALPHA_M = [0.1, 0.3, 0.5, 0.7, 0.9]
        GAMMA = [0.1, 0.3, 0.5, 0.7, 0.9]
        BETA = 1

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        transition_log_headers = get_transition_log_headers()

        for MODEL in MODELS:
            
            # For mb_learnt, alpha_td and alpha_m are independent; for all other models they are identical
            if MODEL == "mb_learnt":
                alpha_combinations = [(a_td, a_m) for a_td in ALPHA_TD for a_m in ALPHA_M]
            else:
                alpha_combinations = [(a, a) for a in ALPHA_TD]

            for a_td, a_m in alpha_combinations:
            
                for g in GAMMA:
                    
                    model_simulation_results = []

                    for condition in CONDITIONS:
                        print(
                            f"> Simulating model {GREEN}{format_model(MODEL)}{RESET} "
                            f"for condition {GREEN}{format_condition(condition)}{RESET} ..."
                        )

                        # Run simulations
                        simulation_results = run_simulations(SIMULATION_MODE, MODEL, condition, NUM_SIMULATIONS, a_td, a_m, BETA, g)

                        # Add to simulation results per model for all conditions
                        model_simulation_results.extend(simulation_results)

                    # Write model simulation results per model to .csv file
                    model_simulation_results_filepath = join(
                        OUTPUT_DIR,
                        f"{MODEL}_nsimulations{NUM_SIMULATIONS}_alpha_td{a_td}_alpha_m{a_m}_beta{BETA}_gamma{g}.csv",
                    )

                    print(f"> Writing transition log to {GREEN}{model_simulation_results_filepath}{RESET} ...")
                    with open(model_simulation_results_filepath, "w") as model_simulation_results_file:
                        model_simulation_results_file.write(transition_log_headers[MODEL])
                        transition_log_lines = flatten(
                            [result.transition_log for result in model_simulation_results]
                        )
                        model_simulation_results_file.writelines(suffix_all(transition_log_lines, "\n"))
                    print("> Done\n")

    if SIMULATION_MODE == "ppc":

        # Set number of simulations per parameter combination
        NUM_SIMULATIONS = 1
        
        # Load fitted parameters per participant and model
        individual_parameters = pd.read_csv(INPUT_FILE)
        participants = individual_parameters["subject"].unique()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        transition_log_headers = get_transition_log_headers()

        for MODEL in MODELS:

            print(
                f"> Simulating model {GREEN}{format_model(MODEL)}{RESET} "
            )

            model_rows = individual_parameters.loc[individual_parameters["agent_type"] == MODEL]

            combined_outfile = join(OUTPUT_DIR, f"{MODEL}_ppc_allparticipants.csv")
            print(f"> Writing combined PPC file to {GREEN}{combined_outfile}{RESET} ...")

            with open(combined_outfile, "w") as out_f:
                # Write csv header
                out_f.write(f"participant_id,{transition_log_headers[MODEL]}")

                for participant in participants:

                    print(
                        f"for participant {GREEN}{format_condition(participant)}{RESET}"
                    )

                    participant_row = model_rows.loc[model_rows["subject"] == participant]
                    if participant_row.empty:
                        print(
                            f"Skipping participant {GREEN}{format_condition(participant)}{RESET}: "
                            f"no fitted parameters for {format_model(MODEL)}"
                        )
                        continue

                    ALPHA_TD = participant_row["alpha_rwq"].iloc[0]
                    ALPHA_M = participant_row["alpha_mt"].iloc[0]
                    GAMMA = participant_row["gamma"].iloc[0]
                    BETA = 1
                    MIX_W = None
                    if MODEL in (
                        "hybrid_mf_redsr_4_randsr_wupdate",
                        "hybrid_mf_randsr_noupdate",
                        "hybrid_mf_randsr_wupdate",
                        "hybrid_mf_mb_learnt",
                    ):
                        MIX_W = float(participant_row["w"].iloc[0])

                    # Buffer rows per participant; write only if all conditions succeed.
                    participant_lines = []
                    participant_failed = False

                    for condition in CONDITIONS:

                        print(
                            f"for condition {GREEN}{format_condition(condition)}{RESET} ..."
                        )

                        try:
                            simulation_results = run_simulations(
                                SIMULATION_MODE, MODEL, condition, NUM_SIMULATIONS, ALPHA_TD, ALPHA_M, BETA, GAMMA, MIX_W
                            )
                        except FloatingPointError as e:
                            print(
                                f"Skipping participant {participant} entirely "
                                f"(failed on condition {format_condition(condition)}): {e}"
                            )
                            participant_failed = True
                            break

                        transition_log_lines = flatten([result.transition_log for result in simulation_results])

                        for line in transition_log_lines:
                            participant_lines.append(f"{participant},{line}\n")

                    if not participant_failed:
                        for line in participant_lines:
                            out_f.write(line)

            print("> Done\n")

    
    
        