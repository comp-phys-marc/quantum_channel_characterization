import functools
from time import time

import jax
import jax.numpy as jnp
import jax.random as jrng
import optax
from matplotlib import pyplot as plt
from tqdm import tqdm

from channels import kraus_channel_as_super_operator, super_operator_to_choi
from non_unitary_circuit import get_non_unitary_matrix_repr

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import numpy as np

NUM_PROBS = 24

HIDDEN_WIDTH = 30
HIDDEN_LAYERS = 2 # TODO change these values
NUM_POLL_OUTPUT = 4


num_wires = 2
num_inputs = 4 ** num_wires
num_input_cuts = 3 ** num_wires
total_inputs = 4 ** num_wires * num_input_cuts

desired_input_shape = num_input_cuts, num_inputs, 1



key = jrng.key(np.random.randint(0, high=10000000))


def turn_to_probs(raw_data): #TODO
    raw_data_squared = raw_data**2
    return  raw_data_squared/ (raw_data_squared.sum() + jnp.finfo(float).eps)


def apply_dropout(hidden_layer_values, dropout_prob):
    if dropout_prob == 0:
        return hidden_layer_values

    shape = hidden_layer_values.shape
    layer = jnp.ones(shape)*1/(1-dropout_prob)
    _, subkey = jrng.split(key)
    layer = layer.at[jrng.choice(subkey, jnp.arange(shape[0]), (int(shape[0] * dropout_prob),))].set(0.)

    return hidden_layer_values * layer


def model_for_probs(input_data, input_weights, hidden_weights, output_weights, dropout_prob=0):
    input_shape = input_data.shape
    input_data_with_bias = jnp.ones((input_shape[0]+1, input_shape[1])).at[:input_shape[0]].set(input_data)
    hidden_layer_1 = jax.vmap(jnp.matmul, in_axes=(0, 1))(input_weights, input_data_with_bias)
    hidden_layer_1 = jnp.apply_along_axis(jax.nn.relu, 0, hidden_layer_1)

    hidden_layer_1_length = num_input_cuts * len(hidden_layer_1[0])
    hidden_layer_values = jnp.reshape(hidden_layer_1, hidden_layer_1_length)
    hidden_layer_values = apply_dropout(hidden_layer_values, dropout_prob)

    for hidden_layer in range(HIDDEN_LAYERS):
        current_len =  hidden_layer_values.shape[0]
        hidden_layer_values_with_bias = jnp.ones((current_len + 1,)).at[:current_len].set(hidden_layer_values)

        hidden_layer_values_with_bias = hidden_weights[hidden_layer] @ hidden_layer_values_with_bias
        hidden_layer_values = jax.nn.relu(hidden_layer_values_with_bias)
        hidden_layer_values = apply_dropout(hidden_layer_values, dropout_prob)

    current_len = hidden_layer_values.shape[0]
    hidden_layer_values_with_bias = jnp.ones((current_len + 1,)).at[:current_len].set(
        hidden_layer_values)

    output_data = output_weights @ hidden_layer_values_with_bias
    setup_error_probs = turn_to_probs(output_data[:4])
    entanglement_error_probs = turn_to_probs(output_data[4:20])
    measurement_error_probs = turn_to_probs(output_data[20:24])

    return  setup_error_probs, entanglement_error_probs, measurement_error_probs


def get_input_weights(num_inputs_per_cut):
    # Array of matrices which are used to poll the input
    required_shape = (num_input_cuts, NUM_POLL_OUTPUT, num_inputs_per_cut + 1)
    return jrng.uniform(key, required_shape)


def get_hidden_weights():
    # List of matrices representing all to all weights for hidden layer
    hidden_width_1 = num_input_cuts * NUM_POLL_OUTPUT
    input_sizes = [hidden_width_1] + ([HIDDEN_WIDTH] * (HIDDEN_LAYERS - 1))
    return  [jrng.uniform(key, (HIDDEN_WIDTH, input_size + 1)) for input_size in input_sizes]

def get_output_weights():
    # Matrices representing all to 24 which returns the raw probs data, (need nonlinear func to make proper probs)
    required_shape = (NUM_PROBS, HIDDEN_WIDTH + 1)
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

# @functools.partial(jax.jit, static_argnames = "dropout_prob")
def batch_loss(params, input_expectations, expected_probs, dropout_prob=0.):
    losses = find_loss(params, input_expectations, expected_probs, dropout_prob)
    return jnp.mean(losses)


def find_loss_choi(params, input_expectations, expected_choi, dropout_prob):
    losses = []
    probs = jax.vmap(model_for_probs, in_axes=(0, None, None, None, None))(input_expectations, *params, dropout_prob)
    reset_probs, cnot_probs, measurement_probs = probs
    for i in range(len(input_expectations)):
        repr = get_non_unitary_matrix_repr(
            2,
            2,
            cnot_probs[i],
            reset_probs[i],
            measurement_probs[i],
            type="tape"
        )
        super_operator = kraus_channel_as_super_operator(repr.unitary_systems, 2)
        choi = super_operator_to_choi(super_operator)
        losses.append(jnp.linalg.norm(choi - expected_choi[i], ord='fro'))
    return jnp.array(losses)


def batch_loss_choi(params, input_expectations, expected_probs, dropout_prob=0.):
    return np.mean(find_loss_choi(params, input_expectations, expected_probs, dropout_prob))

def get_params(num_inputs_per_cut=num_inputs):
    W_input = get_input_weights(num_inputs_per_cut)  # should be constant for now
    W_hidden = get_hidden_weights()
    W_output = get_output_weights()
    return W_input, W_hidden, W_output


def train_model(input_dataset, expected_dataset, num_batches, initial_learing_rate=2e-3, params=None, dropout_prob=0.01, loss_fn=batch_loss, jit=True): # Note, could pull 1 choi at a time
    # Trains the model
    if params is None:
        params = get_params(input_dataset[0].shape[0])
    loss_history = []

    optimizer = optax.adam(initial_learing_rate)
    opt_state = optimizer.init(params)

    if jit:
        loss_fn = jax.jit(loss_fn, static_argnums=(3,))
        grad = jax.jit(jax.grad(loss_fn), static_argnums=(3,))
    else:
        grad = jax.grad(loss_fn)

    start = time()
    for batch_i in tqdm(range(num_batches)):
        input_batch = jnp.array(input_dataset) #TODO check batch size work
        expected_batch = jnp.array(expected_dataset)

        loss_history.append(loss_fn(params, input_batch, expected_batch))
        grads = grad(params, input_batch, expected_batch, dropout_prob)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    end = time()
    print("Time to train: ", end - start, "s")

    return  loss_history, params


@functools.partial(jax.vmap, in_axes=(0, None))
def get_first_layer_set(input_data, W_input):
    input_shape = input_data.shape
    input_data_with_bias = jnp.ones((input_shape[0] + 1, input_shape[1])).at[:input_shape[0]].set(
        input_data)

    hidden_layer_1 = jax.vmap(jnp.matmul, in_axes=(0, 1))(W_input, input_data_with_bias)
    hidden_after_relu = jnp.apply_along_axis(jax.nn.relu, 0, hidden_layer_1)
    return jnp.squeeze(hidden_after_relu)

@functools.partial(jax.vmap, in_axes=(0, None))
def get_first_layer_single(input_data, W_input):
    input_shape = input_data.shape
    input_data_with_bias = jnp.ones((input_shape[0] + 1,)).at[:input_shape[0]].set(
        input_data)

    hidden_layer_1 = W_input @ input_data_with_bias
    hidden_after_relu = jnp.apply_along_axis(jax.nn.relu, 0, hidden_layer_1)
    return jnp.squeeze(hidden_after_relu)

def first_layer_loss(input_data, input_data_weights, W_input, expected_vals):
    # The difference between expected probs and generated probs
    setup_input_data = input_data * input_data_weights
    first_layer_value = get_first_layer_single(setup_input_data, W_input)
    l2_loss = jnp.sum((first_layer_value - expected_vals)**2, -1)
    l1_regression_loss = jnp.sum(jnp.abs(input_data_weights), 0)/len(input_data_weights)
    return  jnp.mean(l2_loss + l1_regression_loss, 0)


def remove_from_input_dataset(input_dataset, removal_indices):
    new_input_dataset = []
    for removal_axis_index, ind in enumerate(removal_indices):
        to_del = input_dataset[:, :, removal_axis_index]
        new_input_dataset.append(jnp.delete(to_del, ind, 1))
    return jnp.transpose(jnp.array(new_input_dataset), (1, 2, 0))


def train_model_lasso_inputs(input_dataset, expected_dataset, num_batches, remove_iterations, initial_learing_rate=2e-3, params=None, dropout_prob=0.05, loss_fn=batch_loss, jit=True): # Note, could pull 1 choi at a time
    removals_per_iteration = 1
    # main_training_batch_size, main_training_num_batches = len(input_dataset), 1500
    _, params = train_model(input_dataset, expected_dataset, num_batches, initial_learing_rate,
                params, dropout_prob, loss_fn, jit=jit) #TODO
    input_dataset = jnp.array(input_dataset)
    first_layer_training_set = get_first_layer_set(input_dataset, params[0])

    optimizer = optax.adam(initial_learing_rate)

    input_data_weights = jnp.ones(input_dataset.shape[1:])
    W_inputs = get_input_weights(num_inputs)
    all_removal_indices = []

    input_dataset = jnp.array(input_dataset)

    for i in tqdm(range(remove_iterations)):
        opt_states = [optimizer.init((input_data_weights[:, inp_cut], W_inputs[inp_cut])) for inp_cut in range(num_input_cuts)]

        grad_fn = jax.vmap(jax.grad(first_layer_loss, argnums=(1, 2)), in_axes=(-1, -1, 0, 1))

        for batch_i in range(num_batches):
            grads = grad_fn(input_dataset, input_data_weights, W_inputs, first_layer_training_set) # TODO
            for inp_cut in range(num_input_cuts):
                updates, opt_states[inp_cut] = optimizer.update((grads[0][inp_cut],grads[1][inp_cut]), opt_states[inp_cut])
                params = optax.apply_updates((input_data_weights[:, inp_cut], W_inputs[inp_cut]), updates)
                input_data_weights = input_data_weights.at[:,inp_cut].set(params[0])
                W_inputs = W_inputs.at[inp_cut].set(params[1])

        # input_data_weights find minimums
        for removal in range(removals_per_iteration):
            removal_indices = jnp.argmin(input_data_weights, 0)
            all_removal_indices.append(removal_indices)
            input_dataset = remove_from_input_dataset(input_dataset, removal_indices)

        # remove said minimums from the input_dataset
        input_data_weights = jnp.ones(input_dataset.shape[1:])
        W_inputs = get_input_weights(num_inputs - (removals_per_iteration * (i+1)))

    params = get_params(num_inputs - (remove_iterations * removals_per_iteration))
    final_loss_history, final_params = train_model(input_dataset, expected_dataset, num_batches, initial_learing_rate,
                params, dropout_prob, loss_fn, jit=jit)

    return final_loss_history, final_params, all_removal_indices


def train_on_full():
    parameters = get_params()
    pre_training_loss = batch_loss(parameters, test_input_data, test_output_data)
    print("pre training loss: ", pre_training_loss)
    loss_hist, parameters = train_model(data["expectations"], data["probs"], 1500,
                                        params=parameters)

    post_training_loss = batch_loss(parameters, test_input_data, test_output_data)
    print("post training loss: ", post_training_loss)

    plt.plot(loss_hist)
    plt.savefig("loss_history.jpg")


def train_with_lasso():
    parameters = get_params()

    pre_training_losses = find_loss(parameters, test_input_data, test_output_data, 0)
    print("pre training loss: ", jnp.mean(pre_training_losses), "+-", jnp.std(pre_training_losses))
    loss_hist, parameters, all_removal_indices = train_model_lasso_inputs(data["expectations"], data["probs"],
                                                                 1500, 5,
                                                                 params=parameters)

    dropped_test_indices = test_input_data

    for removal_indices in all_removal_indices:
        dropped_test_indices = remove_from_input_dataset(dropped_test_indices, removal_indices)

    post_training_losses = find_loss(parameters, dropped_test_indices, test_output_data, 0)
    print("post training loss: ", jnp.mean(post_training_losses), "+-", jnp.std(post_training_losses))

    plt.plot(loss_hist)
    plt.savefig("loss_history_with_lasso.jpg")


def train_with_lasso_choi():
    # data = {"expectations": [], "chois": []}
    #
    # for descriptor, layer in [("", 2), ("additional_", 2), ("two_thousand_", 2)]:
    #     layer_data = jnp.load(f"data/simplified/chois_{descriptor}training_2_qubits_{layer}_layers.npy",
    #                           allow_pickle=True).item()
    #     data["expectations"] += layer_data["expectations"]
    #     data["chois"] += layer_data["chois"]
    #
    # test_data = jnp.load("data/simplified/chois_benchmarking_2_qubits_2_layers.npy",
    #                      allow_pickle=True).item()
    data = {"expectations": [], "chois": []}

    for descriptor, layer in [("", 2)]:
        layer_data = jnp.load(f"data/simplified/chois_{descriptor}training_2_qubits_{layer}_layers.npy",
                              allow_pickle=True).item()
        data["expectations"] += layer_data["expectations"][:5]
        data["chois"] += layer_data["chois"][:5]

    test_data = jnp.load("data/simplified/chois_benchmarking_2_qubits_2_layers.npy",
                         allow_pickle=True).item()

    test_input_data = jnp.array(test_data["expectations"][:5])
    test_output_data = jnp.array(test_data["chois"][:5])

    parameters = get_params()

    pre_training_losses = find_loss_choi(parameters, test_input_data, test_output_data, 0)
    print("pre training loss: ", jnp.mean(pre_training_losses), "+-", jnp.std(pre_training_losses))
    loss_hist, parameters, all_removal_indices = train_model_lasso_inputs(data["expectations"], data["chois"],
                                                                 5, 5,
                                                                 params=parameters, loss_fn=batch_loss_choi, jit=False)

    dropped_test_indices = test_input_data

    for removal_indices in all_removal_indices:
        dropped_test_indices = remove_from_input_dataset(dropped_test_indices, removal_indices)

    post_training_losses = find_loss(parameters, dropped_test_indices, test_output_data, 0, loss_fn=batch_loss_choi)
    print("post training loss: ", jnp.mean(post_training_losses), "+-", jnp.std(post_training_losses))

    plt.plot(loss_hist)
    plt.savefig("loss_history_with_lasso.jpg")


if __name__ == "__main__":
    # train_with_lasso_choi()

    # get error probs
    data = {"expectations": [], "probs": []}
    for descriptor, layer in [("", 2), ("additional_", 2), ("two_thousand_", 2)]:
        layer_data = jnp.load(f"data/simplified/{descriptor}training_2_qubits_{layer}_layers.npy", allow_pickle=True).item()
        data["expectations"] +=  layer_data["expectations"]
        data["probs"] += layer_data["probs"]

    test_data = jnp.load("data/simplified/benchmarking_2_qubits_2_layers.npy", allow_pickle = True).item()
    test_input_data = jnp.array(test_data["expectations"])
    test_output_data = jnp.array(test_data["probs"])

    train_with_lasso()

    # train_on_full()
