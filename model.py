import functools
from time import time

import jax
import jax.numpy as jnp
import jax.random as jrng
import optax

from channels import compare_choi_matrices
import json

NUM_PROBS = 24
NUM_INPUT_CUTS = 8
HIDDEN_WIDTH = 16
HIDDEN_LAYERS = 1 # TODO change these values
NUM_POLL_OUTPUT = 3


num_wires = 4
num_inputs = 4 ** num_wires



key = jrng.key(3112)


def turn_to_probs(raw_data):
    raw_data_squared = raw_data**2
    return  raw_data_squared/ raw_data_squared.sum()


def model_for_probs(input, input_weights, hidden_weights, output_weights):
    reshaped_input = jnp.reshape(input[:,0], (NUM_INPUT_CUTS, int(num_inputs/NUM_INPUT_CUTS)))
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


def get_input_weights(num_inputs_per_cut):
    # Array of matrices which are used to poll the input
    required_shape = (NUM_INPUT_CUTS, NUM_POLL_OUTPUT, num_inputs_per_cut)
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



def get_probs_from_network(params, input_expectations):
    return

@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def find_loss(params, input_expectations, expected_probs):
    # The difference between expected probs and generated probs
    probs_batch = model_for_probs(input_expectations, *params)
    exp_reset_probs = turn_to_probs(expected_probs[:, :4])
    exp_cnot_probs = turn_to_probs(expected_probs[:, 4:20])
    exp_meas_probs = turn_to_probs(expected_probs[:, 20:24])

    loss = 0
    for i, probs in enumerate((exp_reset_probs,exp_cnot_probs, exp_meas_probs)):
        loss += optax.l2_loss(probs_batch[i], probs)/3
    return loss


def batch_loss(params, input_expectations, expected_probs):
    return jnp.mean(find_loss(params, input_expectations, expected_probs))


def train_model(input_dataset, expected_dataset, batch_size, num_batches, initial_learing_rate=1e-2): # Note, could pull 1 choi at a time
    # Trains the model
    W_input = get_input_weights(int(num_inputs/NUM_INPUT_CUTS)) # should be constant for now
    W_hidden = get_hidden_weights()
    W_output = get_output_weights()
    params = W_input, W_hidden, W_output
    loss_history = []

    optimizer = optax.adam(initial_learing_rate)
    opt_state = optimizer.init(params)

    start = time()
    for batch_i in range(num_batches):

        input_batch = jrng.choice(key, jnp.array(input_dataset), (batch_size,)) #TODO check batch size work

        expected_batch = jrng.choice(key, jnp.array(expected_dataset), (batch_size,))

        loss_history.append(batch_loss(params, input_batch, expected_batch))
        grads = jax.grad(batch_loss)(params, input_batch, expected_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    end = time()
    print(end - start)

    qubits, layers = 4, 4
    jnp.save(f"data/params_{qubits}_qubits_{layers}_layers.npy", params)
    return  loss_history


if __name__ == "__main__":
    # get error probs
    data = jnp.load("./data/simplified/training_4_qubits_4_layers.npy", allow_pickle = True).item()

    train_model(data["expectations"], data["probs"], 10, 50)

    # input_dataset = []  # TODO
    # expectation_matrix_from_counts()
    #
    #
    # expected_dataset = []  # TODO
    # loss_history = train_model(input_dataset, expected_dataset, 10, 50)
