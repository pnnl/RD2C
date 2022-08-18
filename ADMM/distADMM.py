import numpy as np
import tensorflow as tf
from mpi4py import MPI
from comm_weights import unflatten_weights, flatten_weights
import os
np.random.seed(132)
tf.keras.backend.set_floatx('float64')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def synthetic_data2d(n, alpha):
    noise_x = alpha*np.random.normal(size=n)
    noise_y = alpha*np.random.normal(size=n)
    x = np.random.uniform(-2*np.pi, 2*np.pi, size=(n, 2))
    y = np.sin(np.cos(x[:, 1]) + noise_x) + np.exp(np.cos(x[:, 0]) + noise_y)
    return x, y


def get_model_architecture(model):
    # find shape and total elements for each layer of the resnet model
    model_weights = model.get_weights()
    layer_shapes = []
    layer_sizes = []
    for i in range(len(model_weights)):
        layer_shapes.append(model_weights[i].shape)
        layer_sizes.append(model_weights[i].size)
    return layer_shapes, layer_sizes


def model_sync(model, layer_shapes, layer_sizes, size):
    # necessary preprocess
    model_weights = model.get_weights()
    # flatten tensor weights
    send_buffer = flatten_weights(model_weights)
    recv_buffer = np.zeros_like(send_buffer)
    # perform all-reduce to synchronize initial models across all clients
    MPI.COMM_WORLD.Allreduce(send_buffer, recv_buffer, op=MPI.SUM)
    # divide by total workers to get average model
    recv_buffer = recv_buffer / size
    # update local models
    new_weights = unflatten_weights(recv_buffer, layer_shapes, layer_sizes)
    model.set_weights(new_weights)


# Implement Custom Loss Function
@tf.function
def consensus_loss(y_true, y_pred, z, lam, rho):
    # local error
    local_mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    # consensus error (lambda term)
    consensus_mse = lam * tf.reduce_sum(y_pred - z)
    # consensus error (rho term)
    consensus_mse += (rho/2)*tf.keras.losses.MeanSquaredError()(z, y_pred)
    return local_mse + consensus_mse


def train(model, lossF, optimizer, train_dataset, coordination_dataset,
          epochs, coord_batch_size, batches, loss_thresh, rho, lam):
    loss_metric = tf.keras.metrics.MeanSquaredError()
    train_loss_metric = tf.keras.metrics.MeanSquaredError()
    for epoch in range(epochs):

        # Local Training
        for batch_idx, (data, target) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_p = model(data, training=True)
                loss_val = lossF(y_true=target, y_pred=y_p)
            grads = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss_metric.update_state(target, y_p)

        # Forward Pass of Coordination Set
        send_predicted = np.zeros((coord_batch_size, batches))
        recv_avg_pred = np.zeros((coord_batch_size, batches))
        for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
            pred = model(c_data, training=True)
            send_predicted[:, c_batch_idx] = pred.numpy().flatten()

        # ADMM Training: 1) Share z 2) Minimize Lagrangian (wrt x) 3) Update lambda

        # Step (1) Share z
        MPI.COMM_WORLD.Allreduce(send_predicted, recv_avg_pred, op=MPI.SUM)
        recv_avg_pred = recv_avg_pred / size

        # Step (2) Minimize Lagrangian
        del_loss = np.Inf
        prev_loss = np.Inf
        if rank == 0:
            print('Minimizing Lagrangian...')
        while del_loss > loss_thresh:
            for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
                with tf.GradientTape() as tape:
                    c_yp = model(c_data, training=True)
                    loss_val = consensus_loss(y_true=c_target, y_pred=c_yp,
                                              z=recv_avg_pred[:, c_batch_idx].reshape(coord_batch_size, 1),
                                              lam=lam, rho=rho)
                grads = tape.gradient(loss_val, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                loss_metric.update_state(c_target, c_yp)
            cur_loss = loss_metric.result()
            del_loss = np.abs(prev_loss - cur_loss)
            prev_loss = cur_loss
            print(cur_loss)
        if rank == 0:
            print('Lagrangian Minimization Threshold Met...')



        # Step (3) Update lambda
        # Second Forward Pass of Coordination Set
        lam_sum = 0
        for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
            pred = model(c_data, training=True)
            lam_sum += np.sum(pred.numpy().flatten() - recv_avg_pred[:, c_batch_idx])
        lam += rho*lam_sum

        if rank == 0:
            print('Rank %d Training Loss for Epoch %d: %0.4f' % (rank, epoch, train_loss_metric.result()))
        train_loss_metric.reset_states()


def run(rank, size):

    # Hyper-parameters
    n = 1000
    alpha = 0.05
    epochs = 50
    learning_rate = 0.0025

    # 2d example
    X, Y = synthetic_data2d(n, alpha)

    # Rescale data between 0 and 1
    data_max = np.max(X)
    data_min = np.min(X)
    X = (X - data_min) / (data_max - data_min)
    # Split up data
    train_split = 0.8
    batch_size = 64
    num_data = len(Y)
    train_x = X[0:int(num_data * train_split), :]
    train_y = Y[0:int(num_data * train_split)]
    test_x = X[int(num_data * train_split):, :]
    test_y = Y[int(num_data * train_split):]
    # convert to tensors
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    # shuffle and batch
    train_dataset = train_dataset.shuffle(int(num_data * train_split)).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Coordination set construction
    coord_size = 160
    # c_batch_size = 16
    c_batch_size = 160
    c_num_batches = int(coord_size / c_batch_size)

    true_x = np.tile(np.linspace(-2 * np.pi, 2 * np.pi, coord_size), (2, 1)).transpose()
    true_y = np.sin(np.cos(true_x[:, 1])) + np.exp(np.cos(true_x[:, 0]))
    coord_max = np.max(true_x)
    coord_min = np.min(true_x)
    true_x = (true_x - coord_min) / (coord_max - coord_min)
    coordination_dataset = tf.data.Dataset.from_tensor_slices((true_x, true_y))
    coordination_dataset = coordination_dataset.batch(c_batch_size)

    # Initialize Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # get model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    # Initialize Local Loss Function
    lossF = tf.keras.losses.MeanSquaredError()

    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model_sync(model, layer_shapes, layer_sizes, size)
    MPI.COMM_WORLD.Barrier()

    # run training
    rho = 0.1
    lam = 0.1
    loss_thresh = 0.1
    train(model, lossF, optimizer, train_dataset, coordination_dataset,
          epochs, c_batch_size, c_num_batches, loss_thresh, rho, lam)


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    run(rank, size)
