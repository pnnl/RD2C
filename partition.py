import tensorflow as tf


def partition_dataset(data, rank, size, batch_size):
    total_data_size = data.cardinality().numpy()
    worker_data_size = total_data_size / size
    data.skip(rank*worker_data_size)
    worker_data = data.take(worker_data_size)
    worker_data = worker_data.shuffle(worker_data_size)
    return worker_data.batch(batch_size)
