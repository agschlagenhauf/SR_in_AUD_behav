#
# likelihood_computation.py
#

'''
LIKELIHOOD COMPUTATION
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utilities import *
from model_update_functions import *

# Initialize random number generator for action selection
rng = np.random.RandomState(42)  # Fixed seed for reproducibility

##### Function to read behavioral data from CSV file #####
def load_phase_data(df):
    """
    Partition a subject's behavioral DataFrame into learning, relearning, and test phases.
    
    Arguments:
        - df: DataFrame containing subject data (e.g. all rows for one condition)
        
    Returns:
        - learning_data: DataFrame with learning phase data (components ending in '-learning')
        - relearning_data: DataFrame with relearning phase data (components ending in '-relearning')
        - test_data: DataFrame with test phase data (components ending in '-test')
    """

    # Filter data by component for robustness
    # Learning phase
    learning_data = df[df['component'].str.endswith('-learning')].copy()
    
    # Relearning phase
    relearning_data = df[df['component'].str.endswith('-relearning')].copy()
    
    # Test phase
    test_data = df[df['component'].str.endswith('-test')].copy()
    
    # Verify we have data
    unique_components = df['component'].unique().tolist()
    if len(learning_data) == 0:
        raise ValueError(f"No learning data found. Check 'component' column values. Available: {unique_components}")
    if len(relearning_data) == 0:
        raise ValueError(f"No relearning data found. Check 'component' column values. Available: {unique_components}")
    if len(test_data) == 0:
        raise ValueError(f"No test data found. Check 'component' column values. Available: {unique_components}")

    return learning_data, relearning_data, test_data

##### Function to parse state and action from behavioral data #####
def parse_state_action(state_str, choice):
    """
    Parses state and action from behavioral data.
    
    Arguments:
        - state_str: state description from CSV
        - choice: choice made (left/right/single/middle) or None for state-only parsing
        
    Returns: (
        - model_state: model state index (0-based integer, or None if invalid)
        - action: action number (0 for left/single/middle, 1 for right, or None if invalid or choice=None)
    )
    """
    # Extract the complete number from state string (the starting state number)
    state = ""
    for char in state_str:
        if char.isdigit():
            state += char  # Build the complete number
        elif state:  # If we've started building a number and hit non-digit, stop
            break
    
    # Convert to model state index (0-based)
    try:
        model_state = int(state) - 1
        if not (0 <= model_state <= 9):  # Check if it's a valid state (1-10)
            return None, None
    except ValueError:
        return None, None
    
    # Convert choice to action index (NaN treated as 0)
    # Handle numeric/NaN choices first
    if choice is None:
        action = 0
    # If choice is a float and NaN, treat as 0
    elif isinstance(choice, float) and np.isnan(choice):
        action = 0
    else:
        choice_str = str(choice).strip().lower()
        if choice_str in ['', 'nan']:
            action = 0
        elif choice_str == 'left':
            action = 0
        elif choice_str == 'right':
            action = 1
        else:
            return None, None
    
    return model_state, action


def compute_trial_likelihood(
    trial_data, 
    v_state, 
    feat, 
    weight,
    t_counts, 
    t_matrix,
    transitions,
    alpha_rwq, 
    alpha_mt, 
    beta, 
    gamma, 
    num_pairs,
    num_states,
    compute_likelihood=True, 
    agent_type=None
    ):
    """
    Computes the likelihood of a single trial and updates model parameters.
    
    Arguments:
        - trial_data: DataFrame with trial data (will be converted to list for indexing)
        - v_state: state-action values
        - feat: successor matrix representing expected future occupancy of state-action pairs (SR only)
        - weight: weight vector learned through TD learning
        - t_counts: transition counts matrix (MB and random-SR learning phase)
        - t_matrix: normalized / learned transition matrix (MB-family agents)
        - transitions: transition structure (used for indexing)
        - alpha_rwq: TD learning rate for weight updates
        - alpha_mt: SR / transition learning rate (SR-like and mb_learnt agents)
        - beta: inverse temperature for choice probabilities
        - gamma: discount factor
        - num_pairs: number of state-action pairs
        - num_states: number of states
        - compute_likelihood: whether to compute log-likelihood (False for non-likelihood trials)
        - agent_type: one of
            'sr', 'mb', 'mb_learnt', 'mf',
            'redsr_2', 'redsr_3', 'redsr_4',
            'randsr_noupdate', 'randsr_wupdate',
            'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'
        
    Returns:
        - log_likelihood: log likelihood of the trial (None if compute_likelihood=False)
        - v_state: updated state-action values
        - feat: updated successor matrix (SR only) or None (MB)
        - weight: updated weight vector
        - t_counts: updated transition counts matrix (MB only) or None (SR)
        - t_matrix: updated normalized transition matrix (MB only) or None (SR)
    """
    
    # Convert trial_data to dictionnary for easier indexing
    trial_list = trial_data.to_dict('records')
    
    # Initialize
    log_likelihood = 0.0
    last_state = None  # Store the string representation of the last state
    last_action = None
    state = None  # Store the string representation of the current state
    action = None
    next_state = None
    next_action = None
    
    # For each state visit in a trial
    for i, row in enumerate(trial_list):

        # Get state and action from row
        state_str = str(row['state'])
        choice_str = row['choice']
        trial_num = row['trial']

        # Processing state and action

        state, action = parse_state_action(state_str, choice_str)
        sa_index = get_flattened_index(transitions, state, action)
        
        ###############################################################################################
        ##### Compute choice probability BEFORE updating the model (only if computing likelihood) #####
        ###############################################################################################
       
        if compute_likelihood:

            # For two-choice states (1-3), compute choice probability
            available_actions = transitions[state]
            if len(available_actions) > 1:

                # Get the starting index for this state (not for the specific action chosen)
                state_start_index = get_flattened_index(transitions, state, 0)
                    
                action_values = v_state[state_start_index:state_start_index + len(available_actions)]
                choice_probs = softmax(beta, action_values)
                choice_prob = choice_probs[action]
                log_likelihood += np.log(max(choice_prob, 1e-10))


        
        ##### Get the actual observed next state from the behavioral data #####
        
        # Check if this is a terminal state
        if i + 1 >= len(trial_list):
            next_state = -1
            next_state_str = "terminal" if state == 9 else "trial_end"
            next_sa_index = -1

        # There is a next row in this trial
        else:
            next_row = trial_list[i + 1]
            next_state_str = str(next_row['state'])
            next_choice_str = str(next_row['choice'])
            
            # Parse next state to get model state index
            next_state, next_action = parse_state_action(next_state_str, next_choice_str)
            next_sa_index = get_flattened_index(transitions, next_state, next_action)
        
        ###################################
        ##### Update model parameters #####
        ###################################

        reward = row['reward']
        
        # Update model using the appropriate function based on agent type

        ##### Don't update model parameters for test trials #####
        if str(trial_data['component'].iloc[0]).endswith('-test'):
            pass

        ##### Successor Representation agent #####
        elif agent_type == 'sr':
            v_state, feat, weight = model_update_sr(v_state, feat, weight, sa_index, next_sa_index, 
                                                  next_state, reward, alpha_rwq, alpha_mt, gamma, 
                                                  num_pairs, transitions, last_state, last_action)

        ##### Model-based agent (count-based transitions) #####
        elif agent_type in ['mb']:
            v_state, weight, t_counts, t_matrix = model_update_mb(v_state, weight, t_counts, t_matrix, 
                                                                 sa_index, next_sa_index, state, next_state, 
                                                                 reward, alpha_rwq, beta, gamma, 
                                                                 num_states, transitions, last_state, last_action)

        ##### Model-based agent with directly learnt transition probabilities #####
        elif agent_type in ['mb_learnt']:
            v_state, weight, t_matrix = model_update_mb_learnt(
                v_state, weight, t_matrix, sa_index, next_sa_index, state, next_state,
                reward, alpha_rwq, alpha_mt, beta, gamma, num_states, transitions,
                last_state, last_action
            )

        ##### Model-free agent #####
        elif agent_type == 'mf':
            v_state = model_update_mf(
                v_state, sa_index, next_sa_index, next_state,
                reward, alpha_rwq, gamma, transitions
            )

        ##### Reduced SR agents #####
        
        # Learning phase
        elif agent_type in ['redsr_2', 'redsr_3', 'redsr_4'] and trial_data['component'].iloc[0].endswith('-learning'):
            v_state, feat, weight = model_update_sr(v_state, feat, weight, sa_index, next_sa_index, 
                                                  next_state, reward, alpha_rwq, alpha_mt, gamma, 
                                                  num_pairs, transitions, last_state, last_action)
        # Re-learning phase
        elif agent_type in ['redsr_2', 'redsr_3', 'redsr_4'] and trial_data['component'].iloc[0].endswith('-relearning'):
            v_state, feat, weight = model_update_sr_wupdate(v_state, feat, weight, sa_index, next_sa_index, 
                                                  next_state, reward, alpha_rwq, alpha_mt, gamma, 
                                                  num_pairs, transitions, last_state, last_action)

        ##### (Reduced) random-policy SR agents #####
        
        # Forced-choice learning trials
        elif agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'] and trial_data['component'].iloc[0].endswith('-learning') and trial_num in range(1, 5):
            v_state, weight, t_counts, t_matrix = model_update_mb(v_state, weight, t_counts, t_matrix, sa_index, next_sa_index, 
                                                  state, next_state, reward, alpha_rwq, beta, gamma, 
                                                  num_states, transitions, last_state, last_action)
            
        # Free-choice learning trials
        elif agent_type in ['randsr_noupdate'] and trial_data['component'].iloc[0].endswith('-learning') and trial_num in range(5, 25):
            # No updates needed for randsr_noupdate - just skip parameter updates
            for k in range(num_pairs):
                v_state[k] = np.sum(weight * feat[k])
        
        elif agent_type in ['randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'] and trial_data['component'].iloc[0].endswith('-learning') and trial_num in range(5, 25):
            v_state, feat, weight = model_update_sr_wupdate(v_state, feat, weight, sa_index, next_sa_index, 
                                                  next_state, reward, alpha_rwq, alpha_mt, gamma, 
                                                  num_pairs, transitions, last_state, last_action)
        
        # Re-learning trials
        elif agent_type in ['randsr_noupdate', 'randsr_wupdate'] and trial_data['component'].iloc[0].endswith('-relearning'):
            v_state, feat, weight = model_update_sr(v_state, feat, weight, sa_index, next_sa_index, 
                                                  next_state, reward, alpha_rwq, alpha_mt, gamma, 
                                                  num_pairs, transitions, last_state, last_action)
            
        elif agent_type in ['redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'] and trial_data['component'].iloc[0].endswith('-relearning'):
            v_state, feat, weight = model_update_sr_wupdate(v_state, feat, weight, sa_index, next_sa_index, 
                                                  next_state, reward, alpha_rwq, alpha_mt, gamma, 
                                                  num_pairs, transitions, last_state, last_action)
        
        # Update tracking variables only if not terminal
        if next_state != -1:  # Not a terminal state (neither state 10 nor trial end)
            last_state = state
            last_action = action
        else:
            # Reset tracking variables at terminal state
            last_state = None
            last_action = None
        
    if compute_likelihood:
        if agent_type in ['sr', 'redsr_2', 'redsr_3', 'redsr_4']:
            return log_likelihood, v_state, feat, weight, None, None
        elif agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'] and trial_data['component'].iloc[0].endswith('-learning') and trial_num in range(1, 5):
            return log_likelihood, v_state, None, weight, t_counts, t_matrix
        elif agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate']:
            return log_likelihood, v_state, feat, weight, None, None
        elif agent_type == 'mb':
            return log_likelihood, v_state, None, weight, t_counts, t_matrix
        elif agent_type == 'mb_learnt':
            return log_likelihood, v_state, None, weight, None, t_matrix
        elif agent_type == 'mf':
            return log_likelihood, v_state, None, None, None, None
    else:
        if agent_type in ['sr', 'redsr_2', 'redsr_3', 'redsr_4']:
            return None, v_state, feat, weight, None, None
        elif agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'] and trial_data['component'].iloc[0].endswith('-learning') and trial_num in range(1, 5):
            return None, v_state, None, weight, t_counts, t_matrix
        elif agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate']:
            return None, v_state, feat, weight, None, None
        elif agent_type == 'mb':
            return None, v_state, None, weight, t_counts, t_matrix
        elif agent_type == 'mb_learnt':
            return None, v_state, None, weight, None, t_matrix
        elif agent_type == 'mf':
            return None, v_state, None, None, None, None


def compute_trial_likelihood_hybrid(
    trial_data,
    v_state_non_mf,
    feat,
    weight,
    t_counts,
    t_matrix,
    v_state_mf,
    transitions,
    alpha_rwq,
    alpha_mt,
    beta,
    gamma,
    mix_w,
    num_pairs,
    num_states,
    compute_likelihood=True,
    agent_type=None
):
    """
    Hybrid trial likelihood: mixture of MF and non-MF choice probabilities (softmax on each branch).
    Non-MF updates depend on agent_type (hybrid_mf_mb_learnt, hybrid_mf_redsr_4_randsr_wupdate, hybrid_mf_randsr_noupdate, hybrid_mf_randsr_wupdate).
    """
    trial_list = trial_data.to_dict('records')

    # Initialize
    log_likelihood = 0.0
    last_state = None  # Store the string representation of the last state
    last_action = None
    state = None  # Store the string representation of the current state
    action = None
    next_state = None
    next_action = None

    for i, row in enumerate(trial_list):
        state_str = str(row['state'])
        choice_str = row['choice']
        trial_num = row['trial']
        component = trial_data['component'].iloc[0]

        state, action = parse_state_action(state_str, choice_str)
        sa_index = get_flattened_index(transitions, state, action)

        if compute_likelihood:
            available_actions = transitions[state]
            if len(available_actions) > 1:
                state_start_index = get_flattened_index(transitions, state, 0)
                # Get the action values for the non MF branch
                action_values_non_mf = v_state_non_mf[state_start_index:state_start_index + len(available_actions)]
                # Get the action values for the MF branch
                action_values_mf = v_state_mf[state_start_index:state_start_index + len(available_actions)]
                # Compute the choice probabilities for the REDSR_4_RANDSR_WUPDATE branch
                choice_probs_non_mf = softmax(beta, action_values_non_mf)
                # Compute the choice probabilities for the MF branch
                choice_probs_mf = softmax(beta, action_values_mf)
                # Compute the choice probability for the hybrid model
                choice_prob = mix_w * choice_probs_non_mf[action] + (1.0 - mix_w) * choice_probs_mf[action]
                log_likelihood += np.log(max(choice_prob, 1e-10))

        if i + 1 >= len(trial_list):
            next_state = -1
            next_state_str = "terminal" if state == 9 else "trial_end"
            next_sa_index = -1

        else:
            next_row = trial_list[i + 1]
            next_state_str = str(next_row['state'])
            next_choice_str = str(next_row['choice'])
            next_state, next_action = parse_state_action(next_state_str, next_choice_str)
            next_sa_index = get_flattened_index(transitions, next_state, next_action)

        reward = row['reward']

        ##### Update MF parameters for all non-test trials #####
        if str(trial_data['component'].iloc[0]).endswith('-test'):
            pass
        else:
            v_state_mf = model_update_mf(
                v_state_mf, sa_index, next_sa_index, next_state, reward, alpha_rwq, gamma, transitions
            )

        ##### Update non-MF parameters for all non-test trials #####
        if str(trial_data['component'].iloc[0]).endswith('-test'):
            pass

        ##### Model-based agent with directly learnt transition probabilities #####
        elif agent_type in ['hybrid_mf_mb_learnt']:
            v_state_non_mf, weight, t_matrix = model_update_mb_learnt(
                v_state_non_mf, weight, t_matrix, sa_index, next_sa_index, state, next_state,
                reward, alpha_rwq, alpha_mt, beta, gamma, num_states, transitions,
                last_state, last_action
            )

        ##### Random-policy Successor Representation agents #####

        # Forced-choice learning trials
        elif agent_type in ['hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_wupdate', 'hybrid_mf_randsr_noupdate'] and component.endswith('-learning') and trial_num in range(1, 5):

            v_state_non_mf, weight, t_counts, t_matrix = model_update_mb(
                v_state_non_mf, weight, t_counts, t_matrix,
                sa_index, next_sa_index, state, next_state, reward,
                alpha_rwq, beta, gamma, num_states, transitions, last_state, last_action
            )

        # Free-choice learning trials
        elif agent_type in ['hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_wupdate'] and component.endswith('-learning') and trial_num in range(5, 25):

            v_state_non_mf, feat, weight= model_update_sr_wupdate(
                v_state_non_mf, feat, weight, sa_index, next_sa_index,
                next_state, reward, alpha_rwq, alpha_mt, gamma,
                num_pairs, transitions, last_state, last_action
            )

        elif agent_type in ['hybrid_mf_randsr_noupdate'] and component.endswith('-learning') and trial_num in range(5, 25):
            # No updates needed for randsr_noupdate - just skip parameter updates
            for k in range(num_pairs):
                v_state_non_mf[k] = np.sum(weight * feat[k])

        # Re-learning trials
        elif agent_type in ['hybrid_mf_redsr_4_randsr_wupdate'] and component.endswith('-relearning'):

            v_state_non_mf, feat, weight = model_update_sr_wupdate(
                v_state_non_mf, feat, weight, sa_index, next_sa_index,
                next_state, reward, alpha_rwq, alpha_mt, gamma,
                num_pairs, transitions, last_state, last_action
            )
        
        elif agent_type in ['hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate'] and component.endswith('-relearning'):

            v_state_non_mf, feat, weight = model_update_sr(
                v_state_non_mf, feat, weight, sa_index, next_sa_index,
                next_state, reward, alpha_rwq, alpha_mt, gamma,
                num_pairs, transitions, last_state, last_action
            )

        if next_state != -1:
            last_state = state
            last_action = action
        else:
            last_state = None
            last_action = None

    return (
        log_likelihood if compute_likelihood else None,
        v_state_non_mf,
        feat,
        weight,
        t_counts,
        t_matrix,
        v_state_mf
    )

def compute_full_likelihood(learning_data, 
                            relearning_data, 
                            test_data, 
                            v_state, 
                            feat, 
                            weight, 
                            t_counts,
                            t_matrix,
                            learning_transitions, 
                            relearning_transitions, 
                            correct_first_action_learning, 
                            condition,
                            alpha_rwq, 
                            alpha_mt, 
                            beta, 
                            gamma, 
                            num_pairs, 
                            num_states,
                            agent_type=None):
    """
    Compute the full likelihood across all phases and return intermediate states.
    
    Arguments:
        - agent_type: one of
            'sr', 'mb', 'mb_learnt', 'mf',
            'redsr_2', 'redsr_3', 'redsr_4',
            'randsr_noupdate', 'randsr_wupdate',
            'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'
        - t_counts: transition counts matrix (MB and random-SR learning phase)
        - t_matrix: normalized / learned transition matrix (MB-family agents)
    
    Returns:
        (total_log_likelihood, kept_indices)
    """
    total_log_likelihood = 0.0

    subject_id = learning_data["ID"].unique()
    
    ########################################################
    ##### Process forced-choice trials (no likelihood) #####
    ########################################################
    
    for trial_num in range(1, 5):
        
        trial_data = learning_data[learning_data['trial'] == trial_num]
        
        _, v_state, feat, weight, t_counts, t_matrix = compute_trial_likelihood(
            trial_data, v_state, feat, weight, t_counts, t_matrix, learning_transitions,
            alpha_rwq, alpha_mt, beta, gamma, num_pairs, num_states,
            compute_likelihood=False, agent_type=agent_type
        )

    # Transform MB matrix to random-policy SR matrix for randsr agents
    if agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate']:
        feat = transform_T_to_random_policy_M(t_matrix, v_state, learning_transitions, gamma)
    
    ########################################################
    ##### Process free-choice trials (likelihood based on these) #######
    ########################################################
    
    for trial_num in range(5, 25):
        
        trial_data = learning_data[learning_data['trial'] == trial_num]
        
        trial_ll, v_state, feat, weight, t_counts, t_matrix = compute_trial_likelihood(
            trial_data, v_state, feat, weight, t_counts, t_matrix, learning_transitions,
            alpha_rwq, alpha_mt, beta, gamma, num_pairs, num_states,
            compute_likelihood=True, agent_type=agent_type
        )
        
        total_log_likelihood += trial_ll

    # If reduced SR: reduce feat and weight
    if agent_type in ['redsr_2', 'redsr_3', 'redsr_4', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate']:
        feat, weight, kept_indices = reduce_weight_and_feat(feat, weight, correct_first_action_learning, condition, agent_type)
    else:
        kept_indices = None
    
    ########################################################
    ##### Process relearning trials (no likelihood) ########
    ########################################################
    
    for trial_num in range(1, 10):
        
        trial_data = relearning_data[relearning_data['trial'] == trial_num]

        _, v_state, feat, weight, t_counts, t_matrix = compute_trial_likelihood(
            trial_data, v_state, feat, weight, t_counts, t_matrix, relearning_transitions,
            alpha_rwq, alpha_mt, beta, gamma, num_pairs, num_states,
            compute_likelihood=False, agent_type=agent_type
        )
    
    ########################################################
    ##### Process test trials (likelihood based on these) ##
    ########################################################
    
    for trial_num in range(1, 6):
        
        trial_data = test_data[test_data['trial'] == trial_num]
        
        trial_ll, v_state, feat, weight, t_counts, t_matrix = compute_trial_likelihood(
            trial_data, v_state, feat, weight, t_counts, t_matrix, relearning_transitions,
            alpha_rwq, alpha_mt, beta, gamma, num_pairs, num_states,
            compute_likelihood=True, agent_type=agent_type
        )
        
        total_log_likelihood += trial_ll
    
    return (total_log_likelihood, kept_indices)

def compute_full_likelihood_hybrid(
    learning_data,
    relearning_data,
    test_data,
    v_state_non_mf,
    feat,
    weight,
    t_counts,
    t_matrix,
    v_state_mf,
    learning_transitions,
    relearning_transitions,
    correct_first_action_learning,
    condition,
    alpha_rwq,
    alpha_mt,
    beta,
    gamma,
    mix_w,
    num_pairs,
    num_states,
    agent_type=None
):
    """
    Full likelihood for MF hybrid agents:
    hybrid_mf_redsr_4_randsr_wupdate, hybrid_mf_randsr_noupdate, hybrid_mf_randsr_wupdate, hybrid_mf_mb_learnt.
    Returns (total_log_likelihood, kept_indices); kept_indices set only for the RedSR-4 hybrid.
    """
    total_log_likelihood = 0.0
    kept_indices = None

    ########################################################
    ##### Process forced-choice trials (no likelihood) #####
    ########################################################

    for trial_num in range(1, 5):

        trial_data = learning_data[learning_data['trial'] == trial_num]

        _, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf = compute_trial_likelihood_hybrid(
            trial_data, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf,
            learning_transitions, alpha_rwq, alpha_mt, beta, gamma, mix_w, num_pairs, num_states,
            compute_likelihood=False, agent_type=agent_type
        )
    # Transform MB matrix to random-policy SR matrix for randsr agents
    if agent_type in ['hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate']:
        feat = transform_T_to_random_policy_M(t_matrix, v_state_non_mf, learning_transitions, gamma)

    ########################################################
    ##### Process free-choice trials (likelihood based on these) #######
    ########################################################

    for trial_num in range(5, 25):

        trial_data = learning_data[learning_data['trial'] == trial_num]

        trial_ll, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf = compute_trial_likelihood_hybrid(
            trial_data, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf,
            learning_transitions, alpha_rwq, alpha_mt, beta, gamma, mix_w, num_pairs, num_states,
            compute_likelihood=True, agent_type=agent_type
        )

        total_log_likelihood += trial_ll

    if agent_type in ['hybrid_mf_redsr_4_randsr_wupdate']:
        feat, weight, kept_indices = reduce_weight_and_feat(
            feat, weight, correct_first_action_learning, condition, 'redsr_4_randsr_wupdate'
        )

    ########################################################
    ##### Process relearning trials (no likelihood) ########
    ########################################################

    for trial_num in range(1, 10):

        trial_data = relearning_data[relearning_data['trial'] == trial_num]
        
        _, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf = compute_trial_likelihood_hybrid(
            trial_data, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf,
            relearning_transitions, alpha_rwq, alpha_mt, beta, gamma, mix_w, num_pairs, num_states,
            compute_likelihood=False, agent_type=agent_type
        )

    ########################################################
    ##### Process test trials (likelihood based on these) ##
    ########################################################

    for trial_num in range(1, 6):
        trial_data = test_data[test_data['trial'] == trial_num]
        trial_ll, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf = compute_trial_likelihood_hybrid(
            trial_data, v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf,
            relearning_transitions, alpha_rwq, alpha_mt, beta, gamma, mix_w, num_pairs, num_states,
            compute_likelihood=True, agent_type=agent_type
        )
        total_log_likelihood += trial_ll

    return (
        total_log_likelihood, kept_indices
    )


def run_parameter_estimation_lbfgsb(subject_data, num_pairs, num_states, agent_type='sr'):
    """
    Run parameter estimation using scipy.optimize.minimize.
    
    Arguments:
        - subject_data: DataFrame with all rows for a single subject (all conditions)
        - num_pairs: number of state-action pairs
        - num_states: number of states
        - agent_type: one of
            'sr', 'mb', 'mb_learnt',
            'redsr_2', 'redsr_3', 'redsr_4',
            'randsr_noupdate', 'randsr_wupdate',
            'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate',
            'hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate', 'hybrid_mf_mb_learnt'
        
    Returns:
        - best_params: [alpha_rwq, alpha_mt, beta, gamma, mix_w] - best parameters for this agent
        - best_log_likelihood: best log likelihood achieved
        - kept_indices: indices of columns kept during reduction (for redsr_2/3/4 agents)
        - evaluations: number of function evaluations performed
    """
    
    # Get components
    components = list(subject_data['component'].unique())
    
    # Get conditions
    conditions = []
    for component in components:
        condition, irrelevant = component.rsplit("-", 1)
        if condition not in conditions:
            conditions.append(condition)
    
    global kept_indices_global
    kept_indices_global = None
    
    def objective_function(params):
        """Objective function to minimize (negative log likelihood)."""
        
        # Initialize subject log lik
        subject_log_likelihood = 0.0
        
        if agent_type == 'hybrid_mf_mb_learnt':
            # MF + mb_learnt hybrid: alpha_rwq, alpha_mt, gamma, mix_w (beta fixed)
            alpha_rwq, alpha_mt, gamma, mix_w = params
            beta = 1.0
        elif agent_type in ('hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate'):
            # Same minimal parameterization: shared SR/MB learning rate, gamma, mixture weight w
            alpha_rwq, gamma, mix_w = params
            alpha_mt = alpha_rwq
            beta = 1.0
        elif agent_type in ['mb', 'mf']:
            # MB agents only optimize alpha_rwq, gamma (no alpha_mt)
            alpha_rwq, gamma = params
            alpha_mt = None  # Fixed value for MB agents
            beta = 1 # Fixed value for inv temp
        elif agent_type == 'mb_learnt':
            # MB_learnt agents optimize alpha_rwq, alpha_mt, gamma
            alpha_rwq, alpha_mt, gamma = params
            beta = 1 # Fixed value for inv temp
        else:
            # SR and other agents optimize all parameters
            alpha_rwq, gamma = params
            alpha_mt = alpha_rwq
            beta = 1 # Fixed value for inv temp
            
        for condition in conditions:

            #print(f"    Processing condition {condition}")
            
            condition_data = subject_data[subject_data['component'].str.startswith(condition + '-')]
            
            learning_data, relearning_data, test_data = load_phase_data(condition_data)
            
            # Determine correct first action for each phase
            correct_first_action_learning = learning_data['correct_first_state_action'].iloc[0]
            correct_first_action_relearning = relearning_data['correct_first_state_action'].iloc[0]
            
            # Get transition structures
            learning_transitions = get_learning_transitions()
            relearning_transitions = get_relearning_transitions(correct_first_action_relearning, condition)
        
            if agent_type in ('hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate'):

                # Initialize model structures for the non-MF branch
                v_state_non_mf, feat, weight, t_counts, t_matrix = initialize_model_structures(
                    'redsr_4_randsr_wupdate', num_pairs, num_states
                )

                # Initialize model structures for the MF branch
                v_state_mf, _, _, _, _ = initialize_model_structures('mf', num_pairs, num_states)

                log_likelihood, kept_indices = compute_full_likelihood_hybrid(
                    learning_data, relearning_data, test_data,
                    v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf,
                    learning_transitions, relearning_transitions, correct_first_action_learning, condition,
                    alpha_rwq, alpha_mt, beta, gamma, mix_w, num_pairs, num_states,
                    agent_type=agent_type
                )

            elif agent_type == 'hybrid_mf_mb_learnt':

                # Initialize model structures for the non-MF branch
                v_state_non_mf, feat, weight, t_counts, t_matrix = initialize_model_structures(
                    'mb_learnt', num_pairs, num_states
                )

                # Initialize model structures for the MF branch
                v_state_mf, _, _, _, _ = initialize_model_structures('mf', num_pairs, num_states)

                log_likelihood, kept_indices = compute_full_likelihood_hybrid(
                    learning_data, relearning_data, test_data,
                    v_state_non_mf, feat, weight, t_counts, t_matrix, v_state_mf,
                    learning_transitions, relearning_transitions, correct_first_action_learning, condition,
                    alpha_rwq, alpha_mt, beta, gamma, mix_w, num_pairs, num_states,
                    agent_type=agent_type
                )

            else:
                # Initialize model structures using unified function
                v_state, feat, weight, t_counts, t_matrix = initialize_model_structures(
                    agent_type, num_pairs, num_states
                )
                
                # Compute likelihood with current parameters
                log_likelihood, kept_indices = compute_full_likelihood(
                    learning_data, relearning_data, test_data, v_state, feat, weight,
                    t_counts, t_matrix,
                    learning_transitions, relearning_transitions, correct_first_action_learning, condition,
                    alpha_rwq, alpha_mt, beta, gamma, num_pairs, num_states,
                    agent_type=agent_type
                )
            
            subject_log_likelihood += log_likelihood
        
            # Store kept_indices globally for return
            global kept_indices_global
            kept_indices_global = kept_indices
        
        return -subject_log_likelihood  # Minimize negative log likelihood
    
    # Initial parameter values and bounds depend on agent type:
    # - mb:        estimate alpha_rwq and gamma
    # - mb_learnt: estimate alpha_rwq, alpha_mt and gamma
    # - others:    estimate alpha_rwq and gamma (with alpha_mt tied to alpha_rwq)
    if agent_type == 'hybrid_mf_mb_learnt':
        # [alpha_rwq, alpha_mt, gamma, w]
        initial_params = [0.5, 0.5, 0.5, 0.5]
        bounds = [(0.001, 1.0), (0.001, 1.0), (0.001, 1.0), (0.001, 1.0)]
    elif agent_type in ('hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate'):
        # [alpha_rwq, gamma, w]
        initial_params = [0.5, 0.5, 0.5]
        bounds = [(0.001, 1.0), (0.001, 1.0), (0.001, 1.0)]
    elif agent_type == 'mb_learnt':
        # [alpha_rwq, alpha_mt, gamma]
        initial_params = [0.5, 0.5, 0.5]
        bounds = [(0.001, 1.0), (0.001, 1.0), (0.001, 1.0)]
    else:
        # [alpha_rwq, gamma]
        initial_params = [0.5, 0.5]
        bounds = [(0.001, 1.0), (0.001, 1.0)]

    # L-BFGS-B options: more function evaluations to reduce ABNORMAL_TERMINATION
    options = {"maxfun": 15000}

    # Run optimization
    result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B', options=options)
    
    if result.success:
        # Reconstruct full parameter vector [alpha_rwq, alpha_mt, beta, gamma]
        if agent_type == 'hybrid_mf_mb_learnt':
            alpha_rwq_opt, alpha_mt_opt, gamma_opt, mix_w_opt = result.x
            beta_opt = 1.0
            best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt, mix_w_opt]
        elif agent_type in ('hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate'):
            alpha_rwq_opt, gamma_opt, mix_w_opt = result.x
            alpha_mt_opt = alpha_rwq_opt
            beta_opt = 1.0
            best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt, mix_w_opt]
        elif agent_type == 'mb_learnt':
            alpha_rwq_opt, alpha_mt_opt, gamma_opt = result.x
            beta_opt = 1.0
            best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt]
        else:
            alpha_rwq_opt, gamma_opt = result.x
            if agent_type in ['mb', 'mf']:
                # MB agents: alpha_mt is not used, beta is fixed to 1
                alpha_mt_opt = None
                beta_opt = 1.0
                best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt]
            else:
                # SR and related agents: alpha_mt = alpha_rwq, beta fixed to 1
                alpha_mt_opt = alpha_rwq_opt
                beta_opt = 1.0
                best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt]

        best_log_likelihood = -result.fun

        evaluations = result.nfev if hasattr(result, 'nfev') else None

    else:
        print(f"Optimization failed: {result.message}")
        # Match success-path `best_params` shape so callers (e.g. run_estimation) label columns correctly.
        if agent_type == 'hybrid_mf_mb_learnt':
            alpha_rwq_opt, alpha_mt_opt, gamma_opt, mix_w_opt = initial_params
            beta_opt = 1.0
            best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt, mix_w_opt]
        elif agent_type in ('hybrid_mf_redsr_4_randsr_wupdate', 'hybrid_mf_randsr_noupdate', 'hybrid_mf_randsr_wupdate'):
            alpha_rwq_opt, gamma_opt, mix_w_opt = initial_params
            alpha_mt_opt = alpha_rwq_opt
            beta_opt = 1.0
            best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt, mix_w_opt]
        elif agent_type == 'mb_learnt':
            alpha_rwq_opt, alpha_mt_opt, gamma_opt = initial_params
            beta_opt = 1.0
            best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt]
        else:
            alpha_rwq_opt, gamma_opt = initial_params
            if agent_type in ['mb', 'mf']:
                alpha_mt_opt = None
                beta_opt = 1.0
                best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt]
            else:
                alpha_mt_opt = alpha_rwq_opt
                beta_opt = 1.0
                best_params = [alpha_rwq_opt, alpha_mt_opt, beta_opt, gamma_opt]
        best_log_likelihood = float('-inf')
        evaluations = None

    return best_params, best_log_likelihood, kept_indices_global, evaluations
