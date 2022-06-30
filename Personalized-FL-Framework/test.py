import tensorflow as tf
import os
import json


# get total number of workers
n_workers = int(os.environ['SLURM_NTASKS'])

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


resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
set_tf_config(resolver)

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(cluster_resolver=resolver)
print('Number of devices to do Multi Worker Mirrored Strategy: {}'.format(strategy.num_replicas_in_sync))



# build multi-worker environment from Slurm variables
#cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(jobs=n_workers, port_base=12345)

# use NCCL communication protocol
#implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
#communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

# declare distribution strategy
#strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver,
#                                                     communication_options=communication_options)

# IMPORTANT: set reduction to NONE so we can do the reduction afterwards and divide by global batch size
loss_function = losses_per_batch = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                             reduction=tf.keras.losses.Reduction.NONE)


@tf.function
def train_step(model, optimizer, loss, train_acc, bs, iterator):  # ------------------- training step

    def step_fn(model, optimizer, loss, train_acc, bs, inputs):  # --------------------------------------- per-GPU step

        x, y = inputs
        with tf.GradientTape() as tape:
            # compute predictions
            predictions = model(x, training=True)

            # compute the loss for each batch between the labels and predictions
            losses_per_batch = loss(y, predictions)

            # sum losses per batch and divide by global batch size
            loss = tf.nn.compute_average_loss(losses_per_batch, global_batch_size=bs)

        # compute and apply gradientsstrategies.
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # compute accuracy
        train_acc.update_state(y, predictions)

        return loss

    # `strategy.run()` invokes step_fn on each GPU
    loss_per_gpu = strategy.run(step_fn, args=(model, optimizer, loss, train_acc, bs, next(iterator),))

    # `strategy.reduce()` reduces given value across GPUs and return result on current device
    return strategy.reduce(tf.distribute.ReduceOp.SUM, loss_per_gpu, axis=None)

