import numpy as np
import tensorflow as tf
from mpi4py import MPI
import os
np.random.seed(132)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def trainOld(model, lossF, optimizer, train_dataset, coordination_dataset, epochs, coord_batch_size, batches):
    loss_metric = tf.keras.metrics.MeanSquaredError()
    for epoch in range(epochs):

        # Adjust learning rate
        set_learning_rate(optimizer, epoch)

        train_vars = model.trainable_variables
        accum_grad1 = [tf.zeros_like(var) for var in train_vars]

        # Local Training
        for batch_idx, (data, target) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_p = model(data, training=True)
                loss_val = lossF(y_true=target, y_pred=y_p)
                loss_metric.update_state(target, y_p)
            grads = tape.gradient(loss_val, train_vars)
            accum_grad1 = [(ag + g) for ag, g in zip(accum_grad1, grads)]

        accum_grad1 = [g/len(train_dataset) for g in accum_grad1]
        # optimizer.apply_gradients(zip(accum_grad, train_vars))

        #'''
        # Forward Pass of Coordination Set
        send_predicted = np.zeros((coord_batch_size, batches), dtype=np.float32)
        recv_avg_pred = np.zeros((coord_batch_size, batches), dtype=np.float32)
        for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
            pred = model(c_data, training=True)
            send_predicted[:, c_batch_idx] = pred.numpy().flatten()

        # Communication Process Here
        MPI.COMM_WORLD.Allreduce(send_predicted, recv_avg_pred, op=MPI.SUM)
        recv_avg_pred = recv_avg_pred/size

        # Consensus Training
        train_vars = model.trainable_variables
        accum_grad2 = [tf.zeros_like(var) for var in train_vars]

        for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
            with tf.GradientTape() as tape:
                c_yp = model(c_data, training=True)
                loss_val = consensus_loss(y_true=c_target, y_pred=c_yp,
                                           z=recv_avg_pred[:, c_batch_idx].reshape(coord_batch_size, 1),
                                           l2=0.1)
            grads = tape.gradient(loss_val, train_vars)
            accum_grad2 = [(ag + g) for ag, g in zip(accum_grad2, grads)]

        accum_grad2 = [g / len(coordination_dataset) for g in accum_grad2]
        avg_grad = [(ag1 + ag2)/2 for ag1, ag2 in zip(accum_grad1, accum_grad2)]

        # finally apply the gradients
        optimizer.apply_gradients(zip(avg_grad, train_vars))
        #'''

        if rank == 0 and epoch % 10 == 0:
            print('(Rank %d) Training Loss for Epoch %d: %0.4f' % (rank, epoch, loss_metric.result()))
        loss_metric.reset_states()