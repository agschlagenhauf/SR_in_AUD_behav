#
# model_based.py
#

'''
IMPLEMENTATION OF A MODEL-BASED VALUE ITERATION AGENT
'''

import numpy as np
import random as rd
from utilities import *

from utilities import *

#
# Run Trial
#
def run_trial(phase, trial_index, gamma, alpha_td, alpha_m, beta, end_state, start_state, forced_choice_switch, forced_choice_trial_order, rewards, transitions, v_state, t_counts, t_matrix, weight):
    '''
    Simulates a single trial, from the given start state until the end state is reached.

    Arguments:
        gamma: the time discounting constant
        alpha_td: the learning rate constant
        alpha_m_ irrelevant parameter for MB
        beta: softmax inverse temperature        
        end_state: terminal state
        start_state: state that the agent starts in
        forced_choice_switch: include forced choice trials during learning or not
        forced_choice_trial_order: random order of forced choice trials in the beginning
        rewards: list of rewards corresponding to each action
        transitions: list of valid transitions from each state
        v_state: the values of state-action pairs
        t_counts: the transition matrix (counts)
        t_matrix: the normalized transition matrix (0-1)
        weight: the weight vector

    Returns: (
        - v_state: the values of state-action pairs updated after the current trial
        - t_counts: the transition matrix (counts) updated after the current trial
        - t_matrix: the normalized transition matrix (0-1) updated after the current trial
        - weight: the weight vector updated after the current trial
    )
    '''
    current_state = start_state - 1
    transition_log_lines = []

    while True:

        ###### First state ######
        if (current_state + 1) == start_state:

            ###### Determine next and second next state ######
            
            # if forced choice trials included in learning phase
            if forced_choice_switch == True:
            
                ## 4 forced-choice trials
                if (trial_index == forced_choice_trial_order[0]):
                    # Determine the next state
                    next_move = 0 # forced left choice
                    next_state = transitions[current_state][next_move] - 1
                    # Determine the second next state
                    second_next_move = 0 # forced left choice
                    second_next_state = transitions[next_state][second_next_move] - 1
                elif (trial_index == forced_choice_trial_order[1]):
                    # Determine the next state
                    next_move = 0  # forced left choice
                    next_state = transitions[current_state][next_move] - 1
                    # Determine the second next state
                    second_next_move = 1  # forced right choice
                    second_next_state = transitions[next_state][second_next_move] - 1
                elif (trial_index == forced_choice_trial_order[2]):
                    # Determine the next state
                    next_move = 1  # forced right choice
                    next_state = transitions[current_state][next_move] - 1
                    # Determine the second next state
                    second_next_move = 0  # forced left choice
                    second_next_state = transitions[next_state][second_next_move] - 1
                elif (trial_index == forced_choice_trial_order[3]):
                    # Determine the next state
                    next_move = 1  # forced right choice
                    next_state = transitions[current_state][next_move] - 1
                    # Determine the second next state
                    second_next_move = 1  # forced right choice
                    second_next_state = transitions[next_state][second_next_move] - 1
                
                ## free-choice trials of other index than 0-3
                else:
                    # Determine the next state
                    next_values = v_state[current_state]
                    if len(next_values) == 1:
                        next_move = 0
                    else:
                        next_choice_probs = softmax(beta, next_values)
                        next_move = rng.choice([0, 1], p=next_choice_probs)
                    next_state = transitions[current_state][next_move] - 1 
    
                    # Determine the second next state
                    second_next_values = v_state[next_state]
                    if len(second_next_values) == 1:
                        second_next_move = 0
                    else:
                        second_next_choice_probs = softmax(beta, second_next_values)
                        second_next_move = rng.choice([0, 1], p=second_next_choice_probs)
                    second_next_state = transitions[next_state][second_next_move] - 1  # get second next state
            
            # if no forced choice trials during learning
            else:
                ## free-choice trials for all indices
                # Determine the next state
                next_values = v_state[current_state]
                if len(next_values) == 1:
                    next_move = 0
                else:
                    next_choice_probs = softmax(beta, next_values)
                    next_move = rng.choice([0, 1], p=next_choice_probs)
                next_state = transitions[current_state][next_move] - 1 

                # Determine the second next state
                second_next_values = v_state[next_state]
                if len(second_next_values) == 1:
                    second_next_move = 0
                else:
                    second_next_choice_probs = softmax(beta, second_next_values)
                    second_next_move = rng.choice([0, 1], p=second_next_choice_probs)
                second_next_state = transitions[next_state][second_next_move] - 1  # get second next state

            ###### No update of transition matrix in first state, as we did not transition from anywhere ######

            ###### Update weights with TD learning ######
            reward = rewards[current_state][next_move]
            
# =============================================================================
#             weight_delta = reward + gamma * v_state[next_state][second_next_move] - \
#                            v_state[current_state][next_move]  # get weight prediction error
#             weight[current_state][next_move] += alpha_td * weight_delta  # update weight
# =============================================================================
            
            weight_delta = reward - weight[current_state][next_move]  # get weight prediction error
            weight[current_state][next_move] += alpha_td * weight_delta  # update weight

            ###### Update values of all state-action pairs (Bellman Equation) ######
            # vector of values per state under a given softmax policy (multiplies value of each action available from a state with its probability of being chosen and sums over all actions per state)
            next_state_values = [np.sum(v_state[state] * softmax(beta, v_state[state])) for state in range(len(rewards))]
            for i in range(len(rewards)):
                for j in range(len(rewards[i])):
                    # multiply transition probability from s to s' (all other = 0) by value of s'
                    v_state[i][j] = weight[i][j] + gamma * np.sum(t_matrix[get_flattened_index(rewards, i, j)] * next_state_values)

            ###### Fill in transition log line ######
            transition_log_lines.append(
                f"{current_state + 1},{next_move + 1},{reward},{weight_delta},{comma_separate(flatten(v_state))},{comma_separate(flatten(weight))},{comma_separate(flatten(t_matrix))}"
            )

            ###### Move to the next state ######
            last_state = current_state
            last_move = next_move
            current_state = next_state
            next_move = second_next_move
            next_state = second_next_state

        ###### Middle states ######
        elif (current_state + 1) != end_state:

            ###### Determine second next state ######
            # Next state determined in last state visit
            # Determine second next state
            second_next_values = v_state[next_state]
            if len(second_next_values) == 1:
                second_next_move = 0
            else:
                second_next_choice_probs = softmax(beta, second_next_values)
                second_next_move = rng.choice([0, 1], p=second_next_choice_probs)
            second_next_state = transitions[next_state][second_next_move] - 1

            ###### Update the transition counts row correpsonding to last state and re-normalize transition matrix ######
            t_counts[get_flattened_index(transitions, last_state, last_move), current_state] += 1
            for index, row in enumerate(t_counts):
                if np.sum(row) == 0:
                    t_matrix[index] = np.zeros(len(rewards))
                else:
                    t_matrix[index] = row / np.sum(row)

            ###### Update weights with TD learning ######
            reward = rewards[current_state][next_move]
            
# =============================================================================
#             weight_delta = reward + gamma * v_state[next_state][second_next_move] - \
#                            v_state[current_state][next_move] # get weight prediction error
#             weight[current_state][next_move] += alpha_td * weight_delta # update weight
# =============================================================================
            
            weight_delta = reward - weight[current_state][next_move]  # get weight prediction error
            weight[current_state][next_move] += alpha_td * weight_delta  # update weight
            
            ###### Update values of all state-action pairs (Bellman Equation) ######
            next_state_values = [np.sum(v_state[state] * softmax(beta, v_state[state])) for state in range(len(rewards))]
            for i in range(len(v_state)):
                for j in range(len(v_state[i])):
                    v_state[i][j] = weight[i][j] + gamma * np.sum(t_matrix[get_flattened_index(v_state, i, j)] * next_state_values)

            ###### Fill in transition log line ######
            transition_log_lines.append(
                f"{current_state + 1},{next_move + 1},{reward},{weight_delta},{comma_separate(flatten(v_state))},{comma_separate(flatten(weight))},{comma_separate(flatten(t_matrix))}"
            )

            ###### Move to the next state ######
            last_state = current_state
            last_move = next_move
            current_state = next_state
            next_move = second_next_move
            next_state = second_next_state

        ###### Last state ######
        elif (current_state + 1) == end_state:

            ###### Update the transition counts row correpsonding to last state and re-normalize transition matrix ######
            t_counts[get_flattened_index(rewards, last_state, last_move), current_state] += 1
            for index, row in enumerate(t_counts):
                if np.sum(row) == 0:
                    t_matrix[index] = np.zeros(len(rewards))
                else:
                    t_matrix[index] = row / np.sum(row)

            ###### Update weights with TD learning ######
            reward = rewards[current_state][next_move]
            
# =============================================================================
#             weight_delta = reward - v_state[current_state][next_move]  # get weight prediction error
#             weight[current_state][next_move] += alpha_td * weight_delta  # update weight
# =============================================================================
            
            weight_delta = reward - weight[current_state][next_move]  # get weight prediction error
            weight[current_state][next_move] += alpha_td * weight_delta  # update weight

            ###### Update values of all state-action pairs (Bellman Equation) ######
            next_state_values = [np.sum(v_state[state] * softmax(beta, v_state[state])) for state in range(len(rewards))]
            for i in range(len(v_state)):
                for j in range(len(v_state[i])):
                    v_state[i][j] = weight[i][j] + gamma * np.sum(t_matrix[get_flattened_index(v_state, i, j)] * next_state_values)


            ###### Fill in transition log line ######
            transition_log_lines.append(
                f"{current_state + 1},{next_move + 1},{reward},{weight_delta},{comma_separate(flatten(v_state))},{comma_separate(flatten(weight))},{comma_separate(flatten(t_matrix))}"
            )

            ###### End loop ######
            break

        else:
            assert False

    return v_state, t_counts, t_matrix, weight, transition_log_lines



def learning(gamma, alpha_td, alpha_m, beta, end_state, rewards, transitions, model_parameters, forced_choice_switch):
    '''
    Simulates the learning phase, where the agent has access to the starting state.

    Arguments:
        - gamma: the time discounting constant
        - alpha_td: the learning rate constant
        - alpha_m: irrelevant parameter for MB
        - beta: softmax inverse temperature
        - end_state: terminal state
        - rewards: list of rewards corresponding to each action
        - transitions: list of valid transitions from each state
        - model_parameters: [
            - v_state: the value of state-action pairs
            - t_counts: the transition matrix (should generally be initialized with zeros)
            - t_matrix: the normalized transition matrix (0-1, should generally be initialized with zeros)
            - weight: the weight vector (also generally initialized with zeros)
        - forced_choice_switch: include forced choice trials during learning or not
        ]

    Returns: (
        - model_parameters: [
            - v_state: the value of state-action pairs
            - t_counts: the transition matrix (should generally be initialized with zeros)
            - t_matrix: the normalized transition matrix (0-1, should generally be initialized with zeros)
            - weight: the weight vector (also generally initialized with zeros)
        ]
        - transition_log: [str]
    )
    '''
    ##### initialize model parameters #####
    v_state, t_counts, t_matrix, weight = model_parameters
    phase = "learning"

    ##### Create start states #####
    start_states = np.ones(24, dtype=np.int8)
    
    # Create random forced choice trials order
    forced_choice_trial_order = [0,1,2,3]
    rd.shuffle(forced_choice_trial_order)

    # Run trials
    transition_log = []
    for trial_index, start_state in enumerate(start_states):
        v_state, t_counts, t_matrix, weight, transition_log_lines = run_trial(
            phase,
            trial_index,
            gamma,
            alpha_td,
            alpha_m,
            beta,
            end_state,
            start_state,
            forced_choice_switch,
            forced_choice_trial_order,
            rewards,
            transitions,
            v_state,
            t_counts,
            t_matrix,
            weight
        )

        transition_log_lines = prefix_all(f"{trial_index + 1},", transition_log_lines)
        transition_log.extend(transition_log_lines)

    new_params = [v_state, t_counts, t_matrix, weight]
        
    return new_params, transition_log


def update_parameters(condition, rewards, transitions):
    if condition == "reward":
        rewards = [[0, 0], [0, 0], [0, 0], [0], [0], [0], [45], [0], [30], [0]]
    elif condition == "transition":
        transitions = [[2, 3], [4, 5], [5, 6], [9], [7], [8], [10], [10], [10], [11]]
    elif condition == "policy":
        rewards = [[0, 0], [0, 0], [0, 0], [0], [0], [0], [45], [15], [30], [0]]
    elif condition == "goal":
        rewards = [[0, 0], [0, 0], [0, 0], [45], [0], [0], [15], [0], [30], [0]]
    else:
        rewards = [[0, 0], [0, 0], [0, 0], [0], [0], [0], [15], [0], [45], [0]]

    return rewards, transitions



def relearning(condition, gamma, alpha_td, alpha_m, beta, end_state, rewards, transitions, model_parameters):
    '''
    Simulates the relearning phase, where the agent does not directly experience the starting state.

    Arguments:
        - condition: string representing the relearning condition
        - gamma: the time discounting constant
        - alpha_td: the learning rate constant
        - alpha_m: irrelevant parameter for MB
        - beta: softmax inverse temperature
        - end_state: terminal state
        - rewards: list of rewards corresponding to each action
        - transitions: list of valid transitions from each state
        - model parameters: [
            - v_state: the value of state-action pairs
            - t_counts: the transition matrix, held over from learning
            - t_matrix: the normalized transition matrix, held over from learning
            - weight: the weight vector, held over from learning
        ]

    Returns: (
        - model_parameters: [
            - v_state: the value of state-action pairs
            - t_counts: the transition matrix (should generally be initialized with zeros)
            - t_matrix: the normalized transition matrix (0-1, should generally be initialized with zeros)
            - weight: the weight vector (also generally initialized with zeros)
        ]
        - transition_log: [str]
    )
    '''
    # Unpack logging_lists and model parameters into component variables
    v_state, t_counts, t_matrix, weight = model_parameters
    phase = "relearning"

    # Create start states
    start_states = np.array([4, 4, 4, 5, 5, 5, 6, 6, 6])
    np.random.shuffle(start_states)
    
    forced_choice_switch = False # never starts with forced 2-choice trials, as all trials are 1-choice anyways
    
    # Create empty forced choice trials order to fulfill input of run_trials
    forced_choice_trial_order = []

    transition_log = []
    for trial_index, start_state in enumerate(start_states):
        v_state, t_counts, t_matrix, weight, transition_log_lines = run_trial(
            phase,
            trial_index,
            gamma,
            alpha_td,
            alpha_m,
            beta,
            end_state,
            start_state,
            forced_choice_switch,
            forced_choice_trial_order,
            rewards,
            transitions,
            v_state,
            t_counts,
            t_matrix,
            weight
        )

        transition_log_lines = prefix_all(f"{trial_index + 1},", transition_log_lines)
        transition_log.extend(transition_log_lines)

    new_params = [v_state, t_counts, t_matrix, weight]
        
    return new_params, transition_log

#
# Test Deterministic
#

def test_deterministic(model_parameters):
    
    '''
    Simulates the test phase by comparing the action values of the two possible starting-state actions. The test state action is assumed to always be
    the higher-value choice.

    Arguments:
        - model_parameters: [
            - num_pairs: the number of state-action pairs
            - feat: the successor matrix, held over from learning
            - weight: the weight vector, held over from learning
        ]

    Returns:
        - transition_log: [str]
    '''

    v_state, t_counts, t_matrix, weight = model_parameters

    state1_action_index = np.argmax(v_state[0][0:2])
    state2_action_index = np.argmax(v_state[1][0:2])
    state3_action_index = np.argmax(v_state[2][0:2])

    if state1_action_index == 0:
        state1_action = ACTION_LEFT
    elif state1_action_index == 1:
        state1_action = ACTION_RIGHT
    else:
        raise ValueError
        
    if state2_action_index == 0:
        state2_action = ACTION_LEFT
    elif state2_action_index == 1:
        state2_action = ACTION_RIGHT
    else:
        raise ValueError
        
    if state3_action_index == 0:
        state3_action = ACTION_LEFT
    elif state3_action_index == 1:
        state3_action = ACTION_RIGHT
    else:
        raise ValueError

    # Transition log_line: trial,state,action,reward
    state1_transition_log_line = f"1,1,{state1_action},0"
    if state1_action == ACTION_LEFT:
        stage2_transition_log_line = f"1,2,{state2_action},0"
    elif state1_action == ACTION_RIGHT:
        stage2_transition_log_line = f"1,3,{state3_action},0"
    else:
        raise ValueError

    return [state1_transition_log_line, stage2_transition_log_line]

#
# Test Probabilistic
#

def test_probabilistic(model_parameters, beta):
    
    '''
    Simulates the test phase by comparing the action values of the two possible starting-state actions. The test state action is assumed to always be
    the higher-value choice.

    Arguments:
        - model_parameters: [
            - num_pairs: the number of state-action pairs
            - feat: the successor matrix, held over from learning
            - weight: the weight vector, held over from learning
        ]

    Returns:
        - transition_log: [str]
    '''

    v_state, t_counts, t_matrix, weight = model_parameters

    state1_transition_log_lines = []
    stage2_transition_log_lines = []

    for trial in range(5):

        state1_choice_probs = softmax(beta, v_state[0][0:2])
        state1_action = rng.choice([0, 1], p=state1_choice_probs)

        state2_choice_probs = softmax(beta, v_state[1][0:2])
        state2_action = rng.choice([0, 1], p=state2_choice_probs)

        state3_choice_probs = softmax(beta, v_state[2][0:2])
        state3_action = rng.choice([0, 1], p=state3_choice_probs)

        # Transition log_line: trial,state,action,reward
        state1_transition_log_line = f"{trial+1},1,{state1_action+1},0"
        if state1_action+1 == ACTION_LEFT:
            stage2_transition_log_line = f"{trial+1},2,{state2_action+1},0"
        elif state1_action+1 == ACTION_RIGHT:
            stage2_transition_log_line = f"{trial+1},3,{state3_action+1},0"
        else:
            raise ValueError

        state1_transition_log_lines.append(state1_transition_log_line)
        stage2_transition_log_lines.append(stage2_transition_log_line)

    # Return a flat list of transition log lines so downstream code can treat
    # the result like other agents' `test*()` functions.
    transition_log = []
    for i in range(len(state1_transition_log_lines)):
        transition_log.append(state1_transition_log_lines[i])
        transition_log.append(stage2_transition_log_lines[i])
    return transition_log
