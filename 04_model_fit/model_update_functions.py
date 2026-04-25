#
# model_update_functions.py
#

"""
Model-specific update functions for different reinforcement learning algorithms.

This module contains the core update logic for various RL models, allowing for
easy extension and comparison of different approaches.
"""

import numpy as np
from utilities import get_flattened_index, safe_divide, get_learning_rewards, flatten

##### Unified initialization function for both SR and MB agents #####
def initialize_model_structures(agent_type, num_pairs, num_states):
    """
    Initialize model parameters for SR- and MB-family agents.
    
    Arguments:
        - agent_type: one of
            'sr', 'mb', 'mb_learnt', 'mf',
            'redsr_2', 'redsr_3', 'redsr_4',
            'randsr_noupdate', 'randsr_wupdate',
            'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate'
        - num_pairs: number of state-action pairs (13)
        - num_states: number of states (10)
        
    Returns:
        - v_state: state-action values (flattened array for both agents)
        - feat: successor matrix (SR only, None for MB)
        - weight: weight vector (flattened array for both agents)
        - t_counts: transition counts matrix (MB and random-SR learning phase; None otherwise)
        - t_matrix: normalized / learned transition matrix (MB-family agents; None for pure SR)
    """
    
    if agent_type == 'sr':
        # Successor Representation agent initialization
        
        # v_state: flattened array of state-action values
        v_state = np.zeros(num_pairs)
        
        # feat: successor matrix (num_pairs x num_pairs)
        feat = np.identity(num_pairs)
        
        # weight: flattened weight vector
        weight = np.zeros(num_pairs)
        
        # MB-specific parameters are None for SR agent
        t_counts = None
        t_matrix = None
        
    elif agent_type == 'mb':
        # Model-based agent initialization (count-based transitions)
        
        # v_state: flattened array of state-action values (same as SR for consistency)
        v_state = np.zeros(num_pairs)
        
        # feat: not used in MB agent
        feat = None
        
        # weight: flattened weight vector (same as SR for consistency)
        weight = np.zeros(num_pairs)
        
        # t_counts: transition counts matrix (num_pairs x num_states)
        # Each row represents a state-action pair, each column represents a possible next state
        t_counts = np.zeros((num_pairs, num_states))
        
        # t_matrix: normalized transition probability matrix (num_pairs x num_states)
        # Initially zeros, will be updated as transitions are observed
        t_matrix = np.zeros((num_pairs, num_states))

    elif agent_type == 'mb_learnt':
        # Model-based agent with directly learnt transition probabilities (no counts)
        
        # v_state: flattened array of state-action values
        v_state = np.zeros(num_pairs)
        
        # feat: not used in MB agents
        feat = None
        
        # weight: flattened reward/value weights
        weight = np.zeros(num_pairs)
        
        # t_counts not used; start as None
        t_counts = None
        
        # t_matrix: learned transition probability matrix (num_pairs x num_states)
        t_matrix = np.ones((num_pairs, num_states)) * (1 / num_states)
        t_matrix[num_pairs - 1, :] = 0

    elif agent_type == 'mf':
        # Model-free Q-learning style agent (values only, no explicit SR or transitions)

        # v_state: flattened array of state-action values
        v_state = np.zeros(num_pairs)

        # feat: not used in MF agent
        feat = None
        
        # weight: not used in MF agent
        weight = None

        # t_counts: not used in MF agent
        t_counts = None

        # t_matrix: not used in MF agent
        t_matrix = None
        
    elif agent_type in ['redsr_2', 'redsr_3', 'redsr_4']:
        # Reduced Successor Representation agent initialization (same as SR)
        
        # v_state: flattened array of state-action values
        v_state = np.zeros(num_pairs)
        
        # feat: successor matrix (num_pairs x num_pairs)
        feat = np.identity(num_pairs)
        
        # weight: flattened weight vector
        weight = np.zeros(num_pairs)
        
        # MB-specific parameters are None for RedSR agent
        t_counts = None
        t_matrix = None
        
    elif agent_type in ['randsr_noupdate', 'randsr_wupdate', 'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', 'redsr_4_randsr_wupdate']:
        # Random Successor Representation agent initialization (hybrid MB→SR)
        
        # v_state: flattened array of state-action values
        v_state = np.zeros(num_pairs)
        
        # feat: successor matrix (num_pairs x num_pairs) - will be transformed from MB
        feat = np.identity(num_pairs)
        
        # weight: flattened weight vector
        weight = np.zeros(num_pairs)
        
        # MB-specific parameters needed for initial learning
        t_counts = np.zeros((num_pairs, num_states))
        t_matrix = np.zeros((num_pairs, num_states))
        
    else:
        raise ValueError(
            f"Invalid agent_type: {agent_type}. Must be 'sr', 'mb', 'mb_learnt', 'mf', "
            "'redsr_2', 'redsr_3', 'redsr_4', 'randsr_noupdate', 'randsr_wupdate', "
            "'redsr_2_randsr_wupdate', 'redsr_3_randsr_wupdate', or 'redsr_4_randsr_wupdate'."
        )
    
    return v_state, feat, weight, t_counts, t_matrix

## Transform state-action-state matrix T to state-action-state-action matrix T
def transform_sa_s_to_sa_sa(t_matrix, v_state, transitions, beta):
    """
    Transform a state-action-to-state transition matrix T into a state-action-to-state-action format using a softmax policy.
    
    Arguments:
        - t_matrix: state-action-by-state transition matrix (shape: [num_pairs, num_states])
        - v_state: state-action values (Q-values) as flattened array (shape: [num_pairs])
        - transitions: transition structure to map indices to state-action pairs
        - beta: inverse temperature parameter for softmax
        
    Returns:
        - T_sa_sa: state-action-to-state-action transition matrix (shape: [num_pairs, num_pairs])
    """
    from utilities import softmax, get_flattened_index
    
    num_pairs = t_matrix.shape[0]
    num_states = t_matrix.shape[1]
    
    T_sa_sa = np.zeros((num_pairs, num_pairs))
    
    # For each state-action pair (s, a)
    for sa_from in range(num_pairs):
        # For each possible next state
        for s_prime in range(num_states):
            # Get Q-values for all actions in the next state
            v_next_state = []
            for a_prime in range(len(transitions[s_prime])):
                sa_prime = get_flattened_index(transitions, s_prime, a_prime)
                v_next_state.append(v_state[sa_prime])
            
            # Apply softmax to get action probabilities in the next state
            if len(v_next_state) > 0:
                action_probs = softmax(beta, np.array(v_next_state))
                
                # For each action in the next state
                for a_prime in range(len(transitions[s_prime])):
                    sa_prime = get_flattened_index(transitions, s_prime, a_prime)
                    
                    # Compute state-action-to-state-action transition probability
                    # T(s,a -> s',a') = T(s,a -> s') * π(a'|s')
                    T_sa_sa[sa_from, sa_prime] = t_matrix[sa_from, s_prime] * action_probs[a_prime]
    
    return T_sa_sa

def transform_T_to_random_policy_M(t_matrix, v_state, transitions, gamma):
    """
    Transform a state-action-to-state transition matrix T first into a random-policy state-action-to-state-action matrix T and then into an SR matrix M.
    
    Arguments:
        - t_matrix: state-action-by-state transition matrix (shape: [num_pairs, num_states])
        - v_state: state-action values (Q-values) as flattened array (shape: [num_pairs])
        - transitions: transition structure to map indices to state-action pairs
        - gamma: discount factor used for SR
        
    Returns:
        - random_policy_M: successor representation matrix under random policy
    """
    
    # Initialize parameters
    beta_random = 0  # beta = 0 for random policy (uniform action selection)
    
    # Transform T(s,a -> s') into T(s,a -> s',a') using random policy
    T_sa_sa = transform_sa_s_to_sa_sa(t_matrix, v_state, transitions, beta_random)

    # Transform T(s,a -> s',a') into M(s,a)
    I = np.eye(T_sa_sa.shape[0])  # Identity matrix of size (state-action pairs)
    random_policy_M = np.linalg.inv(I - gamma * T_sa_sa)
    
    return random_policy_M

def transform_T_to_on_policy_M(t_matrix, v_state, transitions, beta, gamma):
    """
    Transform a state-action-to-state transition matrix T into an on-policy SR matrix M.
    
    Arguments:
        - t_matrix: state-action-by-state transition matrix (shape: [num_pairs, num_states])
        - v_state: state-action values (Q-values) as flattened array (shape: [num_pairs])
        - transitions: transition structure to map indices to state-action pairs
        - beta: inverse temperature for the softmax policy
        - gamma: discount factor used for SR
        
    Returns:
        - t_derived_M: successor representation matrix under the on-policy softmax with inverse temperature beta
    """ 
    # Transform T(s,a -> s') into T(s,a -> s',a') using policy
    T_sa_sa = transform_sa_s_to_sa_sa(t_matrix, v_state, transitions, beta)

    # Transform T(s,a -> s',a') into M(s,a)
    I = np.eye(T_sa_sa.shape[0])  # Identity matrix of size (state-action pairs)
    t_derived_M = np.linalg.inv(I - gamma * T_sa_sa)
    
    return t_derived_M

def reduce_weight_and_feat(feat, weight, correct_first_action_learning, condition, agent_type='redsr_2'):
    '''
    Deletes all columns of the successor matrix and the reward vector
    that don't correspond to a reward-giving action, converting a full to a reduced successor matrix

    Arguments:
        - feat: the full successor matrix
        - weight: the weight vector
        - agent_type: the type of agent
        - correct_first_action_learning: the correct first action during learning
        - condition: the condition of the experiment
    Returns:
        - reduced_feat: the reduced successor matrix
        - reduced_weight: the reduced weight vector
        - kept_indices: indices of columns that were kept (for reconstruction)

    '''
    reduced_feat = []
    reduced_weight = []
    kept_indices = []
    
    # Get reward structure (nested by state); flatten to one value per state-action pair (13 pairs)
    rewards_nested = get_learning_rewards(correct_first_action_learning, condition)
    rewards = flatten(rewards_nested)

    for i, w in enumerate(weight):

        if agent_type in ['redsr_2', 'redsr_2_randsr_wupdate']:
            # Keep only state-action pairs with rewards > 0
            if rewards[i] > 0:
                non_zero_feat_column = feat[:,i]
                reduced_feat.append(non_zero_feat_column)
                non_zero_weight = weight[i]
                reduced_weight.append(non_zero_weight)
                kept_indices.append(i)

        elif agent_type in ['redsr_3', 'redsr_3_randsr_wupdate']:
            # Keep state-action pairs with rewards > 0, plus the terminal state (index 12)
            if rewards[i] > 0 or i == 12:
                non_zero_feat_column = feat[:,i]
                reduced_feat.append(non_zero_feat_column)
                non_zero_weight = weight[i]
                reduced_weight.append(non_zero_weight)
                kept_indices.append(i)

        elif agent_type in ['redsr_4', 'redsr_4_randsr_wupdate']:
            if i in [9, 10, 11, 12]:
                non_zero_feat_column = feat[:,i]
                reduced_feat.append(non_zero_feat_column)
                non_zero_weight = weight[i]
                reduced_weight.append(non_zero_weight)
                kept_indices.append(i)

    return np.transpose(reduced_feat), np.array(reduced_weight), kept_indices

##### Function to update SR model parameters #####
def model_update_sr(v_state, feat, weight, sa_index, next_sa_index, next_state, reward, 
                   alpha_rwq, alpha_mt, gamma, num_pairs, transitions, last_state, last_action):
    """
    Update the Successor Representation model parameters.
    
    Arguments:
        - v_state: state-action values
        - feat: successor matrix
        - weight: weight vector
        - sa_index: current state-action index
        - next_sa_index: next state-action index
        - next_state: next state
        - reward: reward received
        - alpha_rwq: TD learning rate
        - alpha_mt: SR learning rate
        - gamma: discount factor
        - num_pairs: number of state-action pairs
        - transitions: transition structure
        - last_state: previous state (for SR matrix update)
        - last_action: previous action (for SR matrix update)
        
    Returns:
        - v_state: updated state-action values
        - feat: updated successor matrix
        - weight: updated weight vector
    """
    
    ##### Update successor matrix #####

    # If there is a previous state
    if last_state is not None and last_action is not None:

        last_sa_index = get_flattened_index(transitions, last_state, last_action)
        
        # Create one-hot vector for last state-action pair
        one_hot = np.zeros(num_pairs)
        one_hot[last_sa_index] = 1

        # TD update for successor matrix: M(s,a) ← M(s,a) + α_M * [1 + γ*M(s',a') - M(s,a)]
        feat_delta = one_hot + gamma * feat[sa_index] - feat[last_sa_index]
        feat[last_sa_index] += alpha_mt * feat_delta
    
    ##### Update weights with TD learning for state-action values #####
    
    # If non-terminal state
    if next_state >= 0 and next_state < len(transitions):
        # TD update: V(s,a) ← V(s,a) + α_TD * [r + γ*V(s',a') - V(s,a)]
        weight_delta = reward + gamma * v_state[next_sa_index] - v_state[sa_index]
    else:
        # Terminal state: V(s,a) ← V(s,a) + α_TD * [r - V(s,a)]
        weight_delta = reward - v_state[sa_index]
    
    # scale feature according to Russek et al. 2017
    feat_norm_squared = np.matmul(feat[sa_index], np.transpose(feat[sa_index]))
    feat_scaled = np.array([safe_divide(feat[sa_index][i], feat_norm_squared) for i in range(len(feat[sa_index]))])

    # Update weight vector
    # w ← w + α_TD * δ * M(s,a) / ||M(s,a)||²
    weight += alpha_rwq * weight_delta * feat_scaled
    
    ##### Update all state-action values #####

    for k in range(num_pairs):
        v_state[k] = np.sum(weight * feat[k])
    
    return v_state, feat, weight

##### Function to update SR model parameters #####
def model_update_sr_wupdate(v_state, feat, weight, sa_index, next_sa_index, next_state, reward, 
                   alpha_rwq, alpha_mt, gamma, num_pairs, transitions, last_state, last_action):
    """
    Update the Successor Representation model parameters.
    
    Arguments:
        - v_state: state-action values
        - feat: successor matrix
        - weight: weight vector
        - sa_index: current state-action index
        - next_sa_index: next state-action index
        - next_state: next state
        - reward: reward received
        - alpha_rwq: TD learning rate
        - alpha_mt: SR learning rate
        - gamma: discount factor
        - num_pairs: number of state-action pairs
        - transitions: transition structure
        - last_state: previous state (for SR matrix update)
        - last_action: previous action (for SR matrix update)
        
    Returns:
        - v_state: updated state-action values
        - feat: updated successor matrix
        - weight: updated weight vector
    """
    
    ##### No update of reduced successor matrix #####
    
    ##### Update weights with TD learning for state-action values #####
    
    # If non-terminal state
    if next_state >= 0 and next_state < len(transitions):
        # TD update: V(s,a) ← V(s,a) + α_TD * [r + γ*V(s',a') - V(s,a)]
        weight_delta = reward + gamma * v_state[next_sa_index] - v_state[sa_index]
    else:
        # Terminal state: V(s,a) ← V(s,a) + α_TD * [r - V(s,a)]
        weight_delta = reward - v_state[sa_index]
    
    # scale feature according to Russek et al. 2017
    feat_norm_squared = np.matmul(feat[sa_index], np.transpose(feat[sa_index]))
    feat_scaled = np.array([safe_divide(feat[sa_index][i], feat_norm_squared) for i in range(len(feat[sa_index]))])

    # Update weight vector
    # w ← w + α_TD * δ * M(s,a) / ||M(s,a)||²
    weight += alpha_rwq * weight_delta * feat_scaled
    
    ##### Update all state-action values #####

    for k in range(num_pairs):
        v_state[k] = np.sum(weight * feat[k])
    
    return v_state, feat, weight


##### Function to update Model-based agent parameters #####
def model_update_mb(v_state, weight, t_counts, t_matrix, sa_index, next_sa_index, state, next_state, reward, 
                   alpha_rwq, beta, gamma, num_states, transitions, last_state, last_action):
    """
    Update the Model-based agent parameters using SR-style flattened indexing.
    
    Arguments:
        - v_state: state-action values (flattened array)
        - weight: weight vector (flattened array)
        - t_counts: transition counts matrix (num_pairs x num_states)
        - t_matrix: normalized transition matrix (num_pairs x num_states)
        - sa_index: current state-action index (flattened)
        - next_sa_index: next state-action index (flattened)
        - state: current state (0-based, passed from likelihood computation)
        - next_state: next state (0-based)
        - reward: reward received
        - alpha_rwq: TD learning rate
        - beta: inverse temperature for action selection
        - gamma: discount factor
        - num_states: number of states
        - transitions: transition structure
        - last_state: previous state (for transition matrix update)
        - last_action: previous action (for transition matrix update)
        
    Returns:
        - v_state: updated state-action values
        - weight: updated weight vector
        - t_counts: updated transition counts matrix
        - t_matrix: updated normalized transition matrix
    """
    
    # Updating MB agent parameters
    
    ##### Update transition matrix (if there is a previous state-action) #####
    
    if last_state is not None and last_action is not None:
        # Updating transition matrix from previous state-action
        
        # Get flattened index for the previous state-action pair
        last_sa_index = get_flattened_index(transitions, last_state, last_action)
        
        # Update transition counts: from last_state to state
        t_counts[last_sa_index, state] += 1
        
        # Re-normalize transition matrix for this state-action pair
        total_transitions = np.sum(t_counts[last_sa_index, :])
        if total_transitions > 0:
            t_matrix[last_sa_index, :] = t_counts[last_sa_index, :] / total_transitions
        else:
            t_matrix[last_sa_index, :] = np.zeros(num_states)
    
    ##### Update reward vector via delta rule #####
    
    weight_delta = reward - weight[sa_index]

    # Update weight vector using flattened indexing
    weight[sa_index] += alpha_rwq * weight_delta
    
    ##### Update all state-action values using Bellman equation #####
    
    from utilities import softmax
    
    # Compute expected state values under current policy
    next_state_values = []
    for s in range(num_states):
        state_action_values = [v_state[get_flattened_index(transitions, s, action)] 
                              for action in range(len(transitions[s]))]
        action_probs = softmax(beta, np.array(state_action_values))
        next_state_values.append(np.sum(np.array(state_action_values) * action_probs))
    
    # Update all state-action values using Bellman equation
    for s in range(num_states):
        for action in range(len(transitions[s])):
            sa_index = get_flattened_index(transitions, s, action)
            v_state[sa_index] = weight[sa_index] + gamma * np.sum(t_matrix[sa_index, :] * next_state_values)
    
    return v_state, weight, t_counts, t_matrix

##### Function to update Model-based agent parameters #####
def model_update_mb_learnt(v_state, weight, t_matrix, sa_index, next_sa_index, state, next_state, reward, 
                   alpha_rwq, alpha_mt, beta, gamma, num_states, transitions, last_state, last_action):
    """
    Update the Model-based agent parameters using SR-style flattened indexing.
    
    Arguments:
        - v_state: state-action values (flattened array)
        - weight: reward/value weight vector (flattened array)
        - t_matrix: learned transition probability matrix (num_pairs x num_states)
        - sa_index: current state-action index (flattened)
        - next_sa_index: next state-action index (flattened)
        - state: current state (0-based, passed from likelihood computation)
        - next_state: next state (0-based)
        - reward: reward received
        - alpha_rwq: TD learning rate for rewards
        - alpha_mt: learning rate for transition probabilities
        - beta: inverse temperature for action selection
        - gamma: discount factor
        - num_states: number of states
        - transitions: transition structure
        - last_state: previous state (for transition matrix update)
        - last_action: previous action (for transition matrix update)
        
    Returns:
        - v_state: updated state-action values
        - weight: updated weight vector
        - t_matrix: updated normalized transition matrix
    """
    
    # Updating MB agent parameters
    
    ##### Update transition matrix (if there is a previous state-action) #####
    
    if last_state is not None and last_action is not None:
        # Updating transition matrix from previous state-action
        
        # Get flattened index for the previous state-action pair
        last_sa_index = get_flattened_index(transitions, last_state, last_action)
        
        one_hot = np.zeros(num_states)
        one_hot[state] = 1
        t_delta = one_hot - t_matrix[last_sa_index]
        t_matrix[last_sa_index] += alpha_mt * t_delta
    
    ##### Update reward vector via delta rule #####
    
    weight_delta = reward - weight[sa_index]

    # Update weight vector using flattened indexing
    weight[sa_index] += alpha_rwq * weight_delta
    
    ##### Update all state-action values using Bellman equation #####
    
    from utilities import softmax
    
    # Compute expected state values under current policy
    next_state_values = []
    for s in range(num_states):
        state_action_values = [v_state[get_flattened_index(transitions, s, action)] 
                              for action in range(len(transitions[s]))]
        action_probs = softmax(beta, np.array(state_action_values))
        next_state_values.append(np.sum(np.array(state_action_values) * action_probs))
    
    # Update all state-action values using Bellman equation
    for s in range(num_states):
        for action in range(len(transitions[s])):
            sa_index = get_flattened_index(transitions, s, action)
            v_state[sa_index] = weight[sa_index] + gamma * np.sum(t_matrix[sa_index, :] * next_state_values)
    
    return v_state, weight, t_matrix

##### Function to update model-free agent parameters #####
def model_update_mf(v_state, sa_index, next_sa_index, next_state, reward,
                   alpha_rwq, gamma, transitions):
    """
    Update a model-free (Q-learning style) agent's state-action values.
    
    Arguments:
        - v_state: state-action values (Q-values), flattened over state-action pairs
        - sa_index: current state-action index
        - next_sa_index: next state-action index
        - next_state: next state (0-based), or -1 for terminal
        - reward: reward received
        - alpha_rwq: TD learning rate
        - gamma: discount factor
        - transitions: transition structure (used only to check valid next_state range)
        
    Returns:
        - v_state: updated state-action values
    """
    ###### Update Q-values with TD learning ######

    # If non-terminal state
    if next_state >= 0 and next_state < len(transitions):
        # TD update: V(s,a) ← V(s,a) + α_TD * [r + γ*V(s',a') - V(s,a)]
        delta = reward + gamma * v_state[next_sa_index] - v_state[sa_index]
    else:
        # Terminal state: V(s,a) ← V(s,a) + α_TD * [r - V(s,a)]
        delta = reward - v_state[sa_index]

    v_state[sa_index] += alpha_rwq * delta
    
    return v_state