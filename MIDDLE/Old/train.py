import tensorflow as tf
import numpy as np
import time


def train(rank, model, lossF, optimizer, train_dataset, epochs, recorder):

    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()

    for epoch in range(epochs):

        # Adjust learning rate
        # set_learning_rate(optimizer, epoch)

        record_time = 0
        e_init_time = time.time()

        # Local Training
        for batch_idx, (data, target) in enumerate(train_dataset):

            # Minibatch Update
            with tf.GradientTape() as tape:
                y_p = model(data, training=True)
                loss_val = lossF(y_true=target, y_pred=y_p)
            grads = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            r_time = time.time()
            acc_metric.update_state(target, y_p)
            loss_metric.update_state(target, y_p)
            record_time += (time.time() - r_time)

        e_time = (time.time() - e_init_time) - record_time
        e_loss = loss_metric.result()
        e_acc = acc_metric.result()
        print('(Rank %d) Epoch %d: Time is %0.4f, Training Accuracy is %0.4f, Loss is %0.4f'
              % (rank, epoch+1, e_time, e_acc, e_loss))
        recorder.add_to_file(e_time, e_time, np.NaN, e_loss, e_acc)
        recorder.save_to_file()
        loss_metric.reset_states()
        acc_metric.reset_states()