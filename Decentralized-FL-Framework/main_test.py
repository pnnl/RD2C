import tensorflow as tf
from mpi4py import MPI
from partition import partition_dataset
from network import Graph
from communication import DecentralizedSGD
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# these work to remove DNN library error
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def run(rank, size):
    epochs = 3
    lr = 0.1
    train_bs = 64
    graph_type = 'ring'

    # inputs = tf.keras.Input(shape=(784,), name="digits")
    # x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    # x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    # outputs = tf.keras.layers.Dense(10, name="predictions")(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Prepare the metrics.
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_metric = tf.keras.metrics.Mean()

    # train_dataset, test_dataset = load_data()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images[-10000:], train_labels[-10000:]))
    train_dataset.batch(train_bs)

    train(model, train_dataset, loss_fn, optimizer, acc_metric, loss_metric, epochs)

    """
    gpus = tf.config.list_logical_devices('GPU')
    num_gpus = len(gpus)
    gpu_id = rank % num_gpus
    assigned_gpu = gpus[gpu_id]
    print(assigned_gpu)

    with tf.device('/device:CPU:0'):
        # load cifar10 data
        train_data, _ = load_data()

        # partition the dataset
        worker_train_data = partition_dataset(train_data, rank, size, train_bs)

    with tf.device(assigned_gpu):

        # initialize model
        # model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)
        model = tf.keras.applications.MobileNetV3Large(include_top=False, weights=None)

        # Use adam optimizer (could use SGD)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

        # Cross entropy loss
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Accuracy metrics
        loss_metric = tf.keras.metrics.Mean()
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    with tf.device(assigned_gpu):
        train(model, worker_train_data, loss_function, optimizer, acc_metric, loss_metric, epochs)
    """


def train(model, train_data, loss_f, optimizer, epoch_accuracy, epoch_loss_avg, epochs):
    training_losses = []
    training_accuracies = []
    comm_times = []
    for epoch in range(epochs):
        comm_time = 0
        for batch_idx, (data, target) in enumerate(train_data):
            print(data.shape)
            # Optimize model
            # loss_value, grads = compute_grad(model, loss_f, data, target)
            with tf.GradientTape() as tape:
                y_p = model(data, training=True)
                loss_val = loss_f(y_true=target, y_pred=y_p)
            grads = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_val)  # Add current batch loss

            # Compare predicted label to actual label
            epoch_accuracy.update_state(target, y_p)

            # with tf.device('/device:CPU:0'):
                # perform model averaging
            #    comm_time += Comm.communicate(model)

            if batch_idx % 50 == 0:
                print('Rank %d Finished Batch %d With Training Loss %0.4f' % (rank, batch_idx, epoch_loss_avg.result()))

        training_losses.append(epoch_loss_avg.result())
        training_accuracies.append(epoch_accuracy.result())
        comm_times.append(comm_time)

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

        epoch_loss_avg.reset_states()
        epoch_accuracy.reset_states()

    return training_accuracies, training_losses


def compute_loss(model, x, y, loss_f, training_bool):
    y_p = model(x, training=training_bool)
    return loss_f(y_true=y, y_pred=y_p)


def compute_grad(model, loss_f, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(model, inputs, targets, loss_f, training_bool=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# REMOVE -10000 AFTER DEBUGGING
def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train[:-10000] / np.float32(255)
  y_train = y_train[:-10000].astype(np.int64)
  x_test = x_test / np.float32(255)
  y_test = y_test.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000) #change back to 50000
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000)
  return train_dataset, test_dataset


if __name__ == "__main__":

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # need to get total num of gpus, then perform a mod operation to get each thread assigned to a gpu index, then
    # for each thread index the gpu list with the given mod index and save that string as its assigned_gpu, finally use
    # the with tf.device(assigned_gpu): to place model and other values onto the assigned gpu

    run(rank, size)
