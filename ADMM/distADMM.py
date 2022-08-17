import numpy as np
import tensorflow as tf
from mpi4py import MPI
# from comm_weights import unflatten_weights, flatten_weights
import os
np.random.seed(132)
tf.keras.backend.set_floatx('float64')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def run(rank, size):
    print(rank)


def train(model, lossF, optimizer, train_dataset, coordination_dataset, epochs, coord_batch_size, batches):
    loss_metric = tf.keras.metrics.MeanSquaredError()
    for epoch in range(epochs):

        # Local Training
        for batch_idx, (data, target) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                y_p = model(data, training=True)
                loss_val = lossF(y_true=target, y_pred=y_p)
            grads = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            loss_metric.update_state(target, y_p)

        # Forward Pass of Coordination Set
        predicted = np.empty((batches, coord_batch_size))
        send_loss = np.empty(batches)
        recv_avg_loss = np.empty(batches)
        for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
            c_yp = model(c_data, training=True)
            predicted[c_batch_idx, :] = c_yp.numpy().flatten()
            send_loss[c_batch_idx] = lossF(y_true=c_target, y_pred=c_yp).numpy()

        # Communication Process Here
        MPI.COMM_WORLD.Allreduce(send_loss, recv_avg_loss, op=MPI.SUM)
        recv_avg_loss = recv_avg_loss/size

        # Consensus Training
        for c_batch_idx, (c_data, c_target) in enumerate(coordination_dataset):
            with tf.GradientTape() as tape:
                c_yp = model(c_data, training=True)
                loss_val = consensus_loss(y_true=c_target, y_pred=c_yp, z=recv_avg_loss[c_batch_idx], l2=1.0)
            grads = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # loss_metric.update_state(c_target, c_yp)

        if rank == 0:
            print('Rank %d Training Loss for Epoch %d: %0.4f' % (rank, epoch, loss_metric.result()))
        loss_metric.reset_states()



if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    run(rank, size)