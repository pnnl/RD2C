import numpy as np
import tensorflow as tf


def partition_dataset(data, rank, size, batch_size):
    total_data_size = data.cardinality().numpy()
    worker_data_size = total_data_size / size
    data.skip(rank*worker_data_size)
    worker_data = data.take(worker_data_size)
    worker_data = worker_data.shuffle(worker_data_size)
    return worker_data.batch(batch_size)


def compute_loss(model, x, y, loss_f, training_bool):
    return loss_f(y_true=y, y_pred=model(x, training=training_bool))


def compute_grad(model, loss_f, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(model, inputs, targets, loss_f, training_bool=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == '__main__':

    # hyper-parameters
    clients = 10
    epochs = 1
    lr = 0.1
    bs = 64

    # initialize dataset
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # split up training data
    client_td = []
    for i in range(clients):
        client_td.append(partition_dataset(train_dataset, i, clients, bs))

    # initialize distributed setup
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope:
        # Initialize random ResNet50 model
        res_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)

        # Use adam optimizer (could use SGD)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07)

        # Cross entropy loss
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # begin training
    for i in range(epochs):
        # if selecting a subset use this code
        # client_order = np.arange(clients)
        # np.random.shuffle(client_order)
        batch = 0
        grad_sum = None
        for j in range(clients):
            train_data = strategy.experimental_distribute_dataset(client_td[j])
            for (data, target) in train_data[batch]:
                loss_value, grads = compute_grad(res_model, loss_function, data, target)

        batch += 1

