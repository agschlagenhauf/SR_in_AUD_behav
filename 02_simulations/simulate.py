# simulate.py
#

import numpy as np
import importlib
from utilities import *

#
# Simulation Result
#
class SimulationResult:

    def __init__(self, transition_log):
        """
        Initializes a SimulationResult.

        Arguments:
            - transition_log: list[str] or str, the rows for one simulation in .csv format
        """

        self.transition_log = transition_log

#
# Run Simulations
#
def run_simulations(simulation_mode, model, condition, num_simulations, alpha_td, alpha_m, beta, gamma, mix_w=None):
    '''
    Runs a number of simulations of a given model, including the learning, relearning, and test phases.

    Arguments:
        - model: str, e.g. "mf", "sr", "redsr_2"
        - condition: str, e.g. "control", "reward", "transition", "policy", "goal"
        - num_simulations: int, number of simulations to run
        - mix_w: optional; mixture weight on non-MF policy for hybrid models (defaults to 0.5 if None)

    Returns:
        - [SimulationResult]: list of simulation results for a (model, condition) pair
    '''

    # Import model module. Models expose:
    # - learning(), update_parameters(), relearning()
    # - test_deterministic() and test_probabilistic() (test-phase logging only)
    model_package = importlib.import_module(model)

    #
    # Seed random number generator
    #
    np.random.seed(42)

    #
    # Initialize default parameters
    #
    
    forced_choice_switch = True
    
    end_state = 10
    num_pairs = 13 # (state, action) pairs
    num_states = 10

    #
    # Simulation loop
    #
    simulation_results = []

    for simulation_number in range(1, num_simulations + 1):
        #
        # Print Simulation Number
        #
        end_character = "\n" if simulation_number == num_simulations else "\r"
        print(f"  > Simulation number: {CYAN}{simulation_number} / {num_simulations}{RESET} ...", end=end_character)

        #
        # Initialize Generic Parameters
        #
        transitions = [[2, 3], [4, 5], [5, 6], [7], [8], [9], [10], [10], [10], [11]]

            
        if condition == "policy" or condition == "transition":
            rewards = [[0, 0], [0, 0], [0, 0], [0], [0], [0], [0], [15], [30], [0]]
        else:
            rewards = [[0, 0], [0, 0], [0, 0], [0], [0], [0], [15], [0], [30], [0]]

        #
        # Initialize Model-Specific Parameters
        #

        ###### model-free ######
        if model == "mf":

            v_state = []

            for j in range(len(rewards)):
                row = []
                for k in range(len(rewards[j])):
                    row.append(0)
                v_state.append(row)

            model_parameters = [v_state]

        ###### model-based ######
        elif model in ["mb"]:
            v_state = []
            init_weight = []

            for j in range(len(rewards)):
                row = []
                for k in range(len(rewards[j])):
                    row.append(0)
                v_state.append(row)
                init_weight.append(row.copy())

            init_t_counts = np.zeros((num_pairs, num_states))
            init_t_matrix = init_t_counts # normalized transition matrix
            model_parameters = [v_state, init_t_counts, init_t_matrix, init_weight]
            
        elif model == "mb_learnt":
            v_state = []
            init_weight = []

            for j in range(len(rewards)):
                row = []
                for k in range(len(rewards[j])):
                    row.append(0)
                v_state.append(row)
                init_weight.append(row.copy())

            init_t_matrix = np.ones((num_pairs, num_states))*(1/num_states) # normalized transition matrix with small non-zero prior
            init_t_matrix[num_states-1, :] = 0 
            model_parameters = [num_states, v_state, init_t_matrix, init_weight]

        ###### full & reduced SR ######
        elif model in ["sr"]:
            v_state = np.zeros(num_pairs)
            init_weight = np.zeros(num_pairs)
            init_sr = np.identity(num_pairs)  # init M with ‚identity matrix as in Russek et al. 2017

            model_parameters = [num_pairs, v_state, init_sr, init_weight]

        elif model in ["randsr_noupdate", 
                       "randsr_wupdate",
                       "redsr_2_randsr_wupdate",
                       "redsr_3_randsr_wupdate",
                       "redsr_4_randsr_wupdate"]:
            v_state = []
            init_weight = []
        
            for j in range(len(rewards)):
                row = []
                for k in range(len(rewards[j])):
                    row.append(0)
                v_state.append(row)
                init_weight.append(row.copy())
        
            init_t_counts = np.zeros((num_pairs, num_states))
            init_t_matrix = init_t_counts # normalized transition matrix
            
            model_parameters = [num_pairs, v_state, init_t_counts, init_t_matrix, init_weight]  

        elif model in ["redsr_2", "redsr_3", "redsr_4"]:
            v_state = np.zeros(num_pairs)
            init_weight = np.zeros(num_pairs)
            init_reduced_weight = np.zeros((num_pairs, 2)) # hard-coded number of columns - adapt!
            init_sr = np.identity(num_pairs)  # init M with identity matrix as in Russek et al. 2017
            init_reduced_sr = np.zeros((num_pairs, 2)) # hard-coded number of columns - adapt!

            model_parameters = [num_pairs, v_state, init_sr, init_reduced_sr, init_weight, init_reduced_weight]

        ###### Hybrid MF models ######
        elif model in ["hybrid_mf_redsr_4_randsr_wupdate", 
                       "hybrid_mf_randsr_noupdate",
                       "hybrid_mf_randsr_wupdate"]:
            v_state_non_mf = []
            v_state_mf = []
            init_weight = []

            for j in range(len(rewards)):
                row = []
                for k in range(len(rewards[j])):
                    row.append(0)
                v_state_non_mf.append(row)
                v_state_mf.append(row.copy())
                init_weight.append(row.copy())

            init_t_counts = np.zeros((num_pairs, num_states))
            init_t_matrix = init_t_counts
            model_parameters = [num_pairs, v_state_non_mf, init_t_counts, init_t_matrix, init_weight, v_state_mf]

        elif model == "hybrid_mf_mb_learnt":
            v_state_non_mf = []
            v_state_mf = []
            init_weight = []

            for j in range(len(rewards)):
                row = []
                for k in range(len(rewards[j])):
                    row.append(0)
                v_state_non_mf.append(row)
                v_state_mf.append(row.copy())
                init_weight.append(row.copy())

            init_t_matrix = np.ones((num_pairs, num_states)) * (1 / num_states)
            init_t_matrix[num_states - 1, :] = 0
            model_parameters = [num_states, v_state_non_mf, init_t_matrix, init_weight, v_state_mf]



        #
        # Learning Phase
        #
        
        if model in (
            "hybrid_mf_redsr_4_randsr_wupdate",
            "hybrid_mf_randsr_noupdate",
            "hybrid_mf_randsr_wupdate",
            "hybrid_mf_mb_learnt",
        ):

            learned_parameters, learning_transition_log = model_package.learning(
                gamma,
                alpha_td,
                alpha_m,
                beta,
                mix_w,
                end_state,
                rewards,
                transitions,
                model_parameters,
                forced_choice_switch,
            )

            learning_test_transition_log = model_package.test_deterministic(learned_parameters, mix_w)

        else:
            learned_parameters, learning_transition_log = model_package.learning(
                gamma,
                alpha_td,
                alpha_m,
                beta,
                end_state,
                rewards,
                transitions,
                model_parameters,
                forced_choice_switch
            )

            learning_test_transition_log = model_package.test_deterministic(learned_parameters)
        
        #
        # Relearning Phase
        #
        new_rewards, new_transitions = model_package.update_parameters(condition, rewards, transitions)

        if model in (
            "hybrid_mf_redsr_4_randsr_wupdate",
            "hybrid_mf_randsr_noupdate",
            "hybrid_mf_randsr_wupdate",
            "hybrid_mf_mb_learnt",
        ):

            relearned_parameters, relearning_transition_log = model_package.relearning(
                condition,
                gamma,
                alpha_td,
                alpha_m,
                beta,
                mix_w,
                end_state,
                new_rewards,
                new_transitions,
                learned_parameters,
            )

        else:

            relearned_parameters, relearning_transition_log = model_package.relearning(
                condition,
                gamma,
                alpha_td,
                alpha_m,
                beta,
                end_state,
                new_rewards,
                new_transitions,
                learned_parameters
            )

        # When simulating from fixed parameters, we simulate deterministic model performance (always choosing the higher-valued option in test trials)
        if simulation_mode == "fixed":
            if model in (
                "hybrid_mf_redsr_4_randsr_wupdate",
                "hybrid_mf_randsr_noupdate",
                "hybrid_mf_randsr_wupdate",
                "hybrid_mf_mb_learnt",
            ):
                relearning_test_transition_log = model_package.test_deterministic(relearned_parameters, mix_w)
            else:
                relearning_test_transition_log = model_package.test_deterministic(relearned_parameters)
        # When simulating from individually fitted parameters, we simulate real-person like behavior that we can later use for parameter recovery
        elif simulation_mode == "ppc":
            if model in (
                "hybrid_mf_redsr_4_randsr_wupdate",
                "hybrid_mf_randsr_noupdate",
                "hybrid_mf_randsr_wupdate",
                "hybrid_mf_mb_learnt",
            ):
                relearning_test_transition_log = model_package.test_probabilistic(relearned_parameters, beta, mix_w)
            else:
                relearning_test_transition_log = model_package.test_probabilistic(relearned_parameters, beta)
       
        #
        # Results
        #
        
        learning_transition_log = prefix_all("learning,", learning_transition_log);
        learning_test_transition_log = prefix_all("learning_test,", learning_test_transition_log);

        relearning_transition_log = prefix_all("relearning,", relearning_transition_log);
        relearning_test_transition_log = prefix_all("relearning_test,", relearning_test_transition_log);

        transition_log = flatten([
            learning_transition_log,
            learning_test_transition_log,
            relearning_transition_log,
            relearning_test_transition_log
        ])

        transition_log = prefix_all(f"{simulation_number},{model},{condition},", transition_log)

        simulation_results.append(
            SimulationResult(
                transition_log=transition_log,
            )
        )

    print(f"  > Done\n")

    # return results for one model, one condition with all phases, all simulations
    return simulation_results


