import functools
from time import time

import jax
import jax.numpy as jnp
import jax.random as jrng
import optax
from matplotlib import pyplot as plt
from tqdm import tqdm
jax.config.update("jax_enable_x64", True)
import numpy as np

NUM_PROBS = 24

HIDDEN_WIDTH = 16
HIDDEN_LAYERS = 1 # TODO change these values
NUM_POLL_OUTPUT = 4


num_wires = 2
num_inputs = 4 ** num_wires
num_input_cuts = 3 ** num_wires
total_inputs = 4 ** num_wires * num_input_cuts

desired_input_shape = num_input_cuts, num_inputs, 1



key = jrng.key(np.random.randint(0, high=10000000))


def turn_to_probs(raw_data):
    raw_data_squared = raw_data**2
    return  raw_data_squared/ raw_data_squared.sum()


def apply_dropout(hidden_layer_values, dropout_prob):
    if dropout_prob == 0:
        return hidden_layer_values

    shape = hidden_layer_values.shape
    layer = jnp.ones(shape)*1/(1-dropout_prob)
    layer = layer.at[jrng.choice(key, jnp.arange(shape[0]), (int(shape[0] * dropout_prob),))].set(0.)

    return hidden_layer_values * layer


def model_for_probs(input_data, input_weights, hidden_weights, output_weights, dropout_prob=0):
    reshaped_input = jnp.reshape(input_data, desired_input_shape)
    # reshaped_input = jax.vmap(apply_dropout, in_axes=(0, None))(reshaped_input, dropout_prob)

    hidden_layer_1 = input_weights @ reshaped_input
    hidden_layer_1 = jnp.apply_along_axis(jax.nn.relu, 0, hidden_layer_1)

    hidden_layer_1_length = num_input_cuts * len(hidden_layer_1[0])
    hidden_layer_values = jnp.reshape(hidden_layer_1, hidden_layer_1_length)
    hidden_layer_values = apply_dropout(hidden_layer_values, dropout_prob)

    for hidden_layer in range(HIDDEN_LAYERS):
        hidden_layer_values = hidden_weights[hidden_layer] @ hidden_layer_values
        hidden_layer_values = jax.nn.relu(hidden_layer_values)
        hidden_layer_values = apply_dropout(hidden_layer_values, dropout_prob)

    output_data = output_weights @ hidden_layer_values
    setup_error_probs = turn_to_probs(output_data[:4])
    entanglement_error_probs = turn_to_probs(output_data[4:20])
    measurement_error_probs = turn_to_probs(output_data[20:24])

    return  setup_error_probs, entanglement_error_probs, measurement_error_probs

def model_for_probs_dynamic_inputs(input_data, input_weights, hidden_weights, output_weights):
    input_poll_index = 0
    hidden_layer_1 = []
    for input_weight_matrix in input_weights:
        end_poll_index = input_poll_index + input_weight_matrix.shape()[1]
        current_input = input_data[input_poll_index: end_poll_index]
        poll_nodes = input_weight_matrix @ current_input
        poll_nodes = jax.nn.relu(poll_nodes)
        hidden_layer_1.append(poll_nodes)

        input_poll_index = end_poll_index

    hidden_layer_1_length = num_input_cuts * NUM_POLL_OUTPUT
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
    required_shape = (num_input_cuts, NUM_POLL_OUTPUT, num_inputs_per_cut)
    return jrng.uniform(key, required_shape)


def get_input_weights_from_list(num_inputs_list):
    # Array of matrices which are used to poll the input
    weights_list = []
    for num_inputs in num_inputs_list:
        weights_list.append(jrng.uniform(key, (NUM_POLL_OUTPUT, num_inputs)))
    return weights_list


def get_hidden_weights():
    # List of matrices representing all to all weights for hidden layer
    hidden_width_1 = num_input_cuts * NUM_POLL_OUTPUT
    input_sizes = [hidden_width_1] + ([HIDDEN_WIDTH] * (HIDDEN_LAYERS - 1))
    return  [jrng.uniform(key, (HIDDEN_WIDTH, input_size)) for input_size in input_sizes]

def get_output_weights():
    # Matrices representing all to 24 which returns the raw probs data, (need nonlinear func to make proper probs)
    required_shape = (NUM_PROBS, HIDDEN_WIDTH)
    return jrng.uniform(key, required_shape)


@functools.partial(jax.vmap, in_axes=(None, 0, 0, None))
def find_loss(params, input_expectations, expected_probs, dropout_prob):
    # The difference between expected probs and generated probs
    probs_batch = model_for_probs(input_expectations, *params, dropout_prob)
    exp_reset_probs = turn_to_probs(expected_probs[:4])
    exp_cnot_probs = turn_to_probs(expected_probs[4:20])
    exp_meas_probs = turn_to_probs(expected_probs[20:24])

    loss = 0
    for i, probs in enumerate((exp_reset_probs,exp_cnot_probs, exp_meas_probs)):
        loss +=jnp.sqrt(jnp.sum((probs_batch[i] - probs)**2))/3
    return loss


def batch_loss(params, input_expectations, expected_probs, dropout_prob=0.):
    return jnp.mean(find_loss(params, input_expectations, expected_probs, dropout_prob))



@functools.partial(jax.vmap, in_axes=(None, 0, 0, None))
def find_loss_choi(params, input_expectations, expected_probs, dropout_prob):
    return
    # The difference between expected probs and generated probs
    # probs_batch = model_for_probs(input_expectations, *params, dropout_prob)
    # exp_reset_probs = turn_to_probs(expected_probs[:4])
    # exp_cnot_probs = turn_to_probs(expected_probs[4:20])
    # exp_meas_probs = turn_to_probs(expected_probs[20:24])
    #
    # loss = 0
    # for i, probs in enumerate((exp_reset_probs,exp_cnot_probs, exp_meas_probs)):
    #     loss +=jnp.sqrt(jnp.sum((probs_batch[i] - probs)**2))/3
    # return loss


def batch_loss_choi(params, input_expectations, expected_probs, dropout_prob=0.):
    return jnp.mean(find_loss(params, input_expectations, expected_probs, dropout_prob))

def get_params(num_inputs_per_cut):
    W_input = get_input_weights(num_inputs)  # should be constant for now
    W_hidden = get_hidden_weights()
    W_output = get_output_weights()
    return W_input, W_hidden, W_output


def train_model(input_dataset, expected_dataset, batch_size, num_batches, initial_learing_rate=1e-2, params=None, dropout_prob=0): # Note, could pull 1 choi at a time
    # Trains the model
    if params is None:
        params = get_params(input_dataset[0].shape[0])
    loss_history = []

    optimizer = optax.adam(initial_learing_rate)
    opt_state = optimizer.init(params)

    start = time()
    for batch_i in tqdm(range(num_batches)):
        input_batch = jrng.choice(key, jnp.array(input_dataset), (batch_size,)) #TODO check batch size work

        expected_batch = jrng.choice(key, jnp.array(expected_dataset), (batch_size,))

        loss_history.append(batch_loss(params, input_batch, expected_batch))
        grads = jax.grad(batch_loss)(params, input_batch, expected_batch, dropout_prob)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    end = time()
    print("Time to train: ", end - start, "s")

    # params_dict = {"W_input": params[0], "W_hidden": params[1], "W_output": params[2]}
    # qubits, layers = 4, 4
    # jnp.save(f"data/params_{qubits}_qubits_{layers}_layers.npy", params_dict)
    return  loss_history, params


@functools.partial(jax.vmap, in_axes=(0, None))
def get_first_layer_set(input_data, W_input):
    reshaped_input = jnp.reshape(input_data, desired_input_shape)

    hidden_layer_1 = W_input @ reshaped_input
    return jnp.apply_along_axis(jax.nn.relu, 0, hidden_layer_1)

def first_layer_loss(input_data, input_data_weights, W_input, expected_vals):
    # The difference between expected probs and generated probs
    setup_input_data = input_data * input_data_weights
    first_layer_value = get_first_layer_set(setup_input_data, W_input)

    return optax.l2_loss(first_layer_value, expected_vals) + jnp.sum(jnp.abs(input_data_weights))/len(input_data_weights)

def train_model_with_loss_run(input_dataset, expected_dataset, batch_size, num_batches, remove_iterations, initial_learing_rate=1e-2, params=None, dropout_prob=0): # Note, could pull 1 choi at a time
    _, params = train_model(input_dataset, expected_dataset, batch_size, num_batches, initial_learing_rate,
                params, dropout_prob)

    first_layer_training_set = jax.vmap(get_first_layer_set, in_axes=(0, None))(input_dataset, params[0])

    optimizer = optax.adam(initial_learing_rate)

    for i in range(remove_iterations):
        if i == 0:
            input_data_weights = jnp.ones(input_dataset.shape[1:])
            W_input = get_input_weights(num_inputs)
        else:
            #input_data_weights find minimums
            #remove said minimums from the input_dataset
            input_data_weights = jnp.ones(input_dataset.shape[1:])
            W_input = get_input_weights(num_inputs-(5*i))

        optimizer = optax.adam(initial_learing_rate)
        opt_state = optimizer.init((input_data_weights, W_input))

        for batch_i in tqdm(range(num_batches)):
            input_batch = jrng.choice(key, jnp.array(input_dataset), (batch_size,)) #TODO check batch size work
            expected_batch = jrng.choice(key, first_layer_training_set, (batch_size,))

            grads = jax.grad(first_layer_loss, argnums=(1, 2))(input_batch, input_data_weights, W_input, expected_batch)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

    return train_model(input_dataset, expected_dataset, batch_size, num_batches, initial_learing_rate,
                params, dropout_prob)


if __name__ == "__main__":
    # get error probs
    data = {"expectations": [], "probs": []}
    for descriptor, layer in [("", 2), ("additional_", 2)]:
        layer_data = jnp.load(f"data/simplified/{descriptor}training_2_qubits_{layer}_layers.npy", allow_pickle=True).item()
        data["expectations"] +=  layer_data["expectations"]
        data["probs"] += layer_data["probs"]

    test_data = jnp.load("data/simplified/benchmarking_2_qubits_2_layers.npy", allow_pickle = True).item()
    parameters = get_params()

    test_input_data = jnp.array(test_data["expectations"])
    test_output_data = jnp.array(test_data["probs"])
    print("pre training loss: ", batch_loss(parameters, test_input_data, test_output_data))
    loss_hist, parameters = train_model(data["expectations"], data["probs"], 100, 1000, params=parameters)
    print("post training loss: ", batch_loss(parameters, test_input_data, test_output_data))


    plt.plot(loss_hist)
    plt.savefig("loss_history.jpg")
