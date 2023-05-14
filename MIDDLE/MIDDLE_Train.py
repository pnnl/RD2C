import tensorflow as tf
import numpy as np
import sys
import time
import copy
sys.path.append('../Decentralized-FL-Framework')
from comm_weights import flatten_weights, unflatten_weights
from tqdm import tqdm


def set_learning_rate(optimizer, epoch):
    if epoch >= 1:
        optimizer.lr = optimizer.lr * tf.math.exp(-0.1)


def average_models(model, local_update, layer_shapes, layer_sizes, lam1, lam2, lam3):
    model_weights = model.get_weights()
    # flatten tensor weights
    coordinate_weights = flatten_weights(model_weights)
    local_weights = flatten_weights(local_update)
    next_weights = unflatten_weights(lam1 * local_weights + (lam2 + lam3) * coordinate_weights, layer_shapes,
                                     layer_sizes)
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
def consensus_step(model, optimizer, c_data, c_target, z, lam2, lam3):
    with tf.GradientTape() as tape:
        c_yp = model(c_data, training=False)
        # local error
        local_loss = lam2 * tf.keras.losses.SparseCategoricalCrossentropy()(c_target, c_yp)
        # consensus error
        consensus_loss = lam3 * tf.keras.losses.CategoricalCrossentropy()(z, c_yp)
        # combined error
        loss_val = local_loss + consensus_loss
    grads = tape.gradient(loss_val, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def middle_train(model, communicator, rank, lossF, optimizer, train_dataset, coordination_x, coordination_y, test_x,
                 test_y, epochs, coord_batch_size, num_outputs, layer_shapes, layer_sizes, recorder,
                 L1=0.5, L2=0.25, L3=0.25):
    
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
        # for (data, target) in tqdm(train_dataset):
        for (data, target) in train_dataset:

            # Forward Pass of Coordination Set (get z)
            send_predicted = np.zeros((num_outputs * coord_batch_size, 1), dtype=np.float32)
            pred = model(coordination_x, training=False)
            send_predicted[:, 0] = pred.numpy().flatten()

            t1 = time.time()
            # Communication Process Here
            recv_avg_pred, c_time = communicator.average(send_predicted)
            comm_time += c_time

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
            z = recv_avg_pred[:, 0].reshape(coord_batch_size, num_outputs)
            consensus_step(model, optimizer, coordination_x, coordination_y, z, L2, L3)

            # update model weights
            average_models(model, local_model, layer_shapes, layer_sizes, L1, L2, L3)


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
