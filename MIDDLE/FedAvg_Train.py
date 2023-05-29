import tensorflow as tf
import sys
import time
import copy
from comm_weights import flatten_weights, unflatten_weights


def set_learning_rate(optimizer, epoch):
    if epoch >= 1:
        optimizer.lr = optimizer.lr * tf.math.exp(-0.1)


def average_models(model, local_update, layer_shapes, layer_sizes, lam1, lam2):
    model_weights = model.get_weights()
    # flatten tensor weights
    coordinate_weights = flatten_weights(model_weights)
    local_weights = flatten_weights(local_update)
    next_weights = unflatten_weights(lam1 * local_weights + lam2 * coordinate_weights, layer_shapes, layer_sizes)
    # update model weights to average
    model.set_weights(next_weights)


# @tf.function
def train_step(model, optimizer, lossF, data, target):
    # Minibatch Update
    with tf.GradientTape() as tape:
        y_p = model(data, training=True)
        loss_val = lossF(y_true=target, y_pred=y_p)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return y_p


# @tf.function
def consensus_step(model, optimizer, lossF, c_data, c_target):
    with tf.GradientTape() as tape:
        c_yp = model(c_data, training=False)
        loss_val = lossF(y_true=c_target, y_pred=c_yp)
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def fedavg_train(model, communicator, rank, lossF, optimizer, train_dataset, coordination_x, coordination_y, test_x,
                 test_y, epochs, layer_shapes, layer_sizes, recorder, L1=0.5, L2=0.5):

    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(int(len(test_y))).batch(1024)
    coordination_x = tf.cast(coordination_x, dtype=tf.float32)

    for epoch in range(epochs):

        # Adjust learning rate
        # set_learning_rate(optimizer, epoch)

        record_time = 0
        comm_time = 0
        non_comp = 0
        e_init_time = time.time()

        # Local Training
        for batch_idx, (data, target) in enumerate(train_dataset):

            t1 = time.time()
            # save pre-batch model
            start_model = copy.deepcopy(model.get_weights())
            non_comp += (time.time() - t1)

            # Minibatch Update
            y_p = train_step(model, optimizer, lossF, data, target)

            t1 = time.time()
            acc_metric.update_state(target, y_p)
            loss_metric.update_state(target, y_p)
            record_time += (time.time() - t1)

            # save model after batch
            local_model = copy.deepcopy(model.get_weights())

            # reset model weights
            model.set_weights(start_model)
            non_comp += (time.time() - t1)

            # Consensus Training
            consensus_step(model, optimizer, lossF, coordination_x, coordination_y)

            # update model weights
            average_models(model, local_model, layer_shapes, layer_sizes, L1, L2)

            # perform FedAvg
            comm_time += communicator.communicate(model)

        e_time = (time.time() - e_init_time) - record_time
        comp_time = e_time - non_comp - comm_time

        e_loss = loss_metric.result()
        e_acc = acc_metric.result()

        # Test Accuracy
        loss_metric.reset_states()
        acc_metric.reset_states()
        for (data, target) in test_dataset:
            y_p = model(data, training=False)
            acc_metric.update_state(target, y_p)
            loss_metric.update_state(target, y_p)

        e_test_loss = loss_metric.result()
        e_test_acc = acc_metric.result()

        print(
            '(Rank %d) Epoch %d: Time is %0.4f, Test Acc: %0.4f, Test Loss: %0.4f, Train Acc: %0.4f, Train Loss: %0.4f'
            % (rank, epoch + 1, e_time, e_test_acc, e_test_loss, e_acc, e_loss))
        recorder.add_to_file(e_time, comp_time, comm_time, e_loss, e_acc, e_test_loss, e_test_acc)
        recorder.save_to_file()
        loss_metric.reset_states()
        acc_metric.reset_states()
