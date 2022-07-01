import tensorflow as tf
import numpy as np
import os
import json


# ---------------------------------------------------------------------------------------
# Code for multi-worker mirrored strategy (each worker has a GPU) on a cluster
# This is meant to be used for a federated experiment where each worker has its own GPU
# ---------------------------------------------------------------------------------------

# get total number of workers
n_workers = int(os.environ['SLURM_NTASKS'])
jobs = {'worker': n_workers}

# Print setup values
print(n_workers)
print("TF version: ", tf.__version__)


def set_tf_config(resolver, environment=None):
    """Set the TF_CONFIG env variable from the given cluster resolver"""
    cfg = {
        'cluster': resolver.cluster_spec().as_dict(),
        'task': {
            'type': resolver.get_task_info()[0],
            'index': resolver.get_task_info()[1],
        },
        'rpc_layer': resolver.rpc_layer,
    }
    if environment:
        cfg['environment'] = environment
    os.environ['TF_CONFIG'] = json.dumps(cfg)


def cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # The `x` arrays are in uint8 and have values in the range [0, 255].
    # You need to convert them to float32 with values in the range [0, 1]
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    x_test = x_test / np.float32(255)
    y_test = y_test.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000)
    return train_dataset, test_dataset


def dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset, _ = cifar10_dataset()
    dataset = dataset.shard(input_context.num_input_pipelines,
                          input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    return dataset


resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(jobs=jobs)
set_tf_config(resolver)

strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=resolver)
print('Number of devices to do Multi Worker Mirrored Strategy: {}'.format(strategy.num_replicas_in_sync))


@tf.function
def train_step(model, optimizer, loss, train_acc, bs, iterator):  # ------------------- training step

    def step_fn(model, optimizer, loss, train_acc, bs, inputs):  # --------------------------------- per-GPU step

        x, y = inputs
        with tf.GradientTape() as tape:
            # compute predictions
            predictions = model(x, training=True)

            # compute the loss for each batch between the labels and predictions
            losses_per_batch = loss(y, predictions)

            # sum losses per batch and divide by global batch size
            loss = tf.nn.compute_average_loss(losses_per_batch, global_batch_size=bs)

        # compute and apply gradients strategies.
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # compute accuracy
        train_acc.update_state(y, predictions)

        return loss

    # `strategy.run()` invokes step_fn on each GPU
    loss_per_gpu = strategy.run(step_fn, args=(model, optimizer, loss, train_acc, bs, next(iterator),))

    # `strategy.reduce()` reduces given value across GPUs and return result on current device
    return strategy.reduce(tf.distribute.ReduceOp.SUM, loss_per_gpu, axis=None)


per_worker_batch_size = 64
global_batch_size = per_worker_batch_size * n_workers

with strategy.scope():
    # Initialize random ResNet50 model
    res_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None)
    # IMPORTANT: set reduction to NONE so we can do the reduction afterwards and divide by global batch size
    loss_function = losses_per_batch = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                            reduction=tf.keras.losses.Reduction.NONE)
    # spread dataset
    multi_worker_dataset = strategy.distribute_datasets_from_function(
        lambda input_context: dataset_fn(global_batch_size, input_context))

    # store accuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

print('Starting Training:')

epochs = 1
iterator = iter(multi_worker_dataset)
num_batches = 70
for epoch in range(epochs):
    total_loss = 0
    for batch in range(num_batches):
        print(batch)
        total_loss += train_step(iterator)

    train_loss = total_loss / num_batches
    print('Epoch: %d, accuracy: %f, train_loss: %f.' % (epoch, train_accuracy.result(), train_loss))
    train_accuracy.reset_states()
