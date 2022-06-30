import tensorflow as tf
from mpi4py import MPI
from partition import partition_dataset
from network import Graph
from communication import DecentralizedSGD
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run(rank, size):
    epochs = 1
    lr = 0.1
    train_bs = 64
    test_bs = 64
    graph_type = 'ring'

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # pre-process training data
    x_train = x_train / 255
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    worker_train_data = partition_dataset(train_dataset, rank, size, train_bs)

    # send out test data
    # x_test = x_test / 255
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # test_dataset = test_dataset.batch(test_bs)

    # Initialize random ResNet50 model
    res_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)

    # find shape and total elements for each layer of the resnet model
    model_weights = res_model.get_weights()
    layer_shapes = []
    layer_sizes = []
    for i in range(len(model_weights)):
        layer_shapes.append(model_weights[i].shape)
        layer_sizes.append(model_weights[i].size)

    Network = Graph(rank, size, MPI.COMM_WORLD, graph_type)
    Communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, Network, layer_shapes, layer_sizes, 0, 1)

    # Use adam optimizer (could use SGD)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07)

    # Cross entropy loss
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Synchronize all models so that initial models are the same
    Communicator.model_sync(res_model)

    MPI.COMM_WORLD.Barrier()

    train(Communicator, res_model, worker_train_data, loss_function, optimizer, epochs)


def train(Comm, model, train_data, loss_f, optimizer, epochs):
    training_losses = []
    training_accuracies = []
    comm_times = []
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        comm_time = 0
        for batch_idx, (data, target) in enumerate(train_data):

            # Optimize model
            loss_value, grads = compute_grad(model, loss_f, data, target)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            # Compare predicted label to actual label
            epoch_accuracy.update_state(target, model(data, training=True))

            # perform model averaging
            comm_time += Comm.communicate(model)

            print('Rank %d Finished Batch %d With Training Loss %0.4f' % (rank, batch_idx, epoch_loss_avg.result()))

        training_losses.append(epoch_loss_avg.result())
        training_accuracies.append(epoch_accuracy.result())
        comm_times.append(comm_time)

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    return training_accuracies, training_losses


def forward_pass(model, x, training_bool):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout)
    return model(x, training=training_bool)


def compute_loss(model, x, y, loss_f, training_bool):
    y_p = forward_pass(model, x, training_bool=training_bool)
    return loss_f(y_true=y, y_pred=y_p)


def compute_grad(model, loss_f, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(model, inputs, targets, loss_f, training_bool=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == "__main__":

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    gpus = tf.config.list_logical_devices('GPU')
    cpus = tf.config.list_logical_devices('CPU')

    # need to get total num of gpus, then perform a mod operation to get each thread assigned to a gpu index, then
    # for each thread index the gpu list with the given mod index and save that string as its assigned_gpu, finally use
    # the with tf.device(assigned_gpu): to place model and other values onto the assigned gpu

    run(rank, size)
