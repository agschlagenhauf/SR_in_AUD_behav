#
# utilities.py
#

import numpy as np

CYAN="\033[36m"
GREEN="\033[32m"
RESET="\033[0m"

ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_FORCED = 1

seed = sum(map(ord, "SR_in_AUD"))
rng = np.random.default_rng(seed)

#def softmax(beta, values):
#    return(np.exp(beta * np.array(values))/np.sum(np.exp(beta * np.array(values))))


def softmax(beta, values):
    """
    Numerically stable softmax over a vector of values.
    
    Arguments:
        - beta: inverse temperature (scales the values before exponentiation)
        - values: array-like of action values
    
    Returns:
        - 1D numpy array of probabilities summing to 1
    """
    v = beta * np.array(values)
    v_stable = v - np.max(v)
    nominator = np.exp(v_stable)
    denominator = np.sum(nominator)
    return nominator / denominator

def safe_divide(numerator, denominator):
    if denominator == 0.0:
        return 0.0
    else:
        return float(numerator) / float(denominator)

def comma_separate(items):
    return ",".join([str(item) for item in items])

def prefix_all(prefix, items):
    return [prefix + item for item in items]

def suffix_all(items, suffix):
    return [item + suffix for item in items]

def format_model(model):
    return model.replace("_", " ").title().replace("Sr", "SR")

def format_condition(condition):
    return condition.replace("_", " ").title()

def flatten(list_input):
    return [item for row in list_input for item in row]

def get_flattened_index(list_input, row, item):
    '''
    Helper function that converts an index of a ragged 2d array into the equivalent index of the flattened array.

    Arguments:
        - list: a 2 dimensional list, which can be ragged
        - row: desired row index of the list
        - item: desired item index of the given row

    Returns: int, the corresponding index in the flattened list
    '''
    index = 0
    for i in range(row):
        index += len(list_input[i])

    index += item

    return index
