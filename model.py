import jax
import jax.numpy as jnp
import jax.random as jrng

NUM_PROBS = 24
NUM_INPUT_CUTS = 8
HIDDEN_WIDTH = 16
HIDDEN_LAYERS = 1 # TODO change these values
NUM_POLL_OUTPUT = 3


num_wires = 4
num_inputs = 4 ** num_wires


input = jnp.array([])

key = jrng.key(3112)


def turn_to_probs(raw_data):
    raw_data_squared = raw_data**2
    return  raw_data_squared/ raw_data_squared.sum()


def model_for_probs(input, input_weights, hidden_weights, output_weights):
    reshaped_input = jnp.reshape(input, (NUM_INPUT_CUTS, num_inputs/NUM_INPUT_CUTS))
    hidden_layer_1 = input_weights @ reshaped_input
    hidden_layer_1 = jnp.apply_along_axis(jax.nn.relu, 0, hidden_layer_1)

    hidden_layer_1_length = NUM_INPUT_CUTS * len(hidden_layer_1[0])
    hidden_layer_values = jnp.reshape(hidden_layer_1, hidden_layer_1_length)

    for hidden_layer in range(HIDDEN_LAYERS):
        hidden_layer_values = hidden_weights[hidden_layer] @ hidden_layer_values
        hidden_layer_values = jax.nn.relu(hidden_layer_values)

    output_data = output_weights @ hidden_layer_values
    setup_error_probs = turn_to_probs(output_data[:4])
    entanglement_error_probs = turn_to_probs(output_data[4:20])
    measurement_error_probs = turn_to_probs(output_data[20:24])

    return  setup_error_probs, entanglement_error_probs, measurement_error_probs

def model_for_probs_dynamic_inputs(input, input_weights, hidden_weights, output_weights):
    input_poll_index = 0
    hidden_layer_1 = []
    for input_weight_matrix in input_weights:
        end_poll_index = input_poll_index + input_weight_matrix.shape()[1]
        current_input = input[input_poll_index: end_poll_index]
        poll_nodes = input_weight_matrix @ current_input
        poll_nodes = jax.nn.relu(poll_nodes)
        hidden_layer_1.append(poll_nodes)

        input_poll_index = end_poll_index

    hidden_layer_1_length = NUM_INPUT_CUTS * NUM_POLL_OUTPUT
    hidden_layer_values = jnp.reshape(hidden_layer_1, hidden_layer_1_length)

    for hidden_layer in range(HIDDEN_LAYERS):
        hidden_layer_values = hidden_weights[hidden_layer] @ hidden_layer_values
        hidden_layer_values = jax.nn.relu(hidden_layer_values)

    output_data = output_weights @ hidden_layer_values
    setup_error_probs = turn_to_probs(output_data[:4])
    entanglement_error_probs = turn_to_probs(output_data[4:20])
    measurement_error_probs = turn_to_probs(output_data[20:24])

    return  setup_error_probs, entanglement_error_probs, measurement_error_probs


def get_input_weights(num_inputs):
    # Array of matrices which are used to poll the input
    required_shape = (NUM_INPUT_CUTS, NUM_POLL_OUTPUT, num_inputs)
    return jrng.uniform(key, required_shape)


def get_input_weights_from_list(num_inputs_list):
    # Array of matrices which are used to poll the input
    weights_list = []
    for num_inputs in num_inputs_list:
        weights_list.append(jrng.uniform(key, (NUM_POLL_OUTPUT, num_inputs)))
    return weights_list


def get_hidden_weights():
    # List of matrices representing all to all weights for hidden layer
    hidden_width_1 = NUM_POLL_OUTPUT * NUM_POLL_OUTPUT
    input_sizes = [hidden_width_1] + ([HIDDEN_WIDTH] * (HIDDEN_LAYERS - 1))
    return  [jrng.uniform(key, (HIDDEN_WIDTH, input_size)) for input_size in input_sizes]

def get_output_weights():
    # Matrices representing all to 24 which returns the raw probs data, (need nonlinear func to make proper probs)
    required_shape = (NUM_PROBS, HIDDEN_WIDTH)
    return jrng.uniform(key, required_shape)

def train_model(): # TODO working on setting up batch training now
    def objective_function(input_weights, hidden_weights, output_weights):
        error_probs = model_for_solving_probs(input, input_weights, hidden_weights, output_weights)
        setup_error_probs, entanglement_error_probs, measurement_error_probs = error_probs
        get_choi_matrix_from_probs(error_probs)
        difference_in_probs ...

        return #The difference between expected choi and non expected choi



    pass # trains the model