import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpi4py import MPI
import argparse
from FedAvg_Train import fedavg_train
from dataloader import darknet_data
from communication import DecentralizedSGD
from misc import Recorder
from network import Graph
import os
import tikzplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_model_architecture(model):
    # find shape and total elements for each layer of the resnet model
    model_weights = model.get_weights()
    layer_shapes = []
    layer_sizes = []
    for i in range(len(model_weights)):
        layer_shapes.append(model_weights[i].shape)
        layer_sizes.append(model_weights[i].size)
    return layer_shapes, layer_sizes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FedAvg Darknet Training')
    parser.add_argument('--name', '-n', default='FedAvg', type=str, help='experiment name')
    parser.add_argument('--experiment', '-exp', default='Darknet', type=str, help='experiment name')
    parser.add_argument('--graph_type', default='ring', type=str, help='baseline topology')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='total epochs')
    parser.add_argument('--bs', default=64, type=int, help='train batch size for each worker')
    parser.add_argument('--train_split', default=0.8, type=float, help='train data percent')
    parser.add_argument('--coord_size', default=128, type=int, help='coordination dataset size')
    parser.add_argument('--L1', default=1/2, type=float, help='train set loss weighting')
    parser.add_argument('--L2', default=1/2, type=float, help='coordination set loss weighting')
    parser.add_argument('--weight_type', default='uniform-neighbor-no-self-weight', type=str, help='worker weightings')
    parser.add_argument('--large_model', default=1, type=int, help='use larger model for fedavg')
    parser.add_argument('--randomSeed', default=482, type=int, help='random seed')

    args = parser.parse_args()

    # initialize random seed
    np.random.seed(args.randomSeed)
    tf.random.set_seed(args.randomSeed + 1)

    mpi = MPI.COMM_WORLD
    bcast = mpi.bcast
    barrier = mpi.barrier
    rank = mpi.rank
    size = mpi.size

    train_pct = args.train_split
    train_bs = args.bs
    test_bs = args.bs
    coordination_size = args.coord_size

    # preprocess and split data amongst workers
    train_set =None; test_set = None; coord_x = None; coord_y = None; nid_train_set = None; nid_test_x = None
    nid_test_y = None; num_inputs = None; num_outputs = None

    if rank == 0:
        print('Beginning Data Preprocessing...')

    if args.experiment == 'Darknet':
        train_set, test_set, coord_x, coord_y, nid_train_set, nid_test_x, nid_test_y, num_inputs, num_outputs = \
            darknet_data(rank, size, train_pct, train_bs, test_bs, coordination_size)

    if rank == 0:
        print('Finished Data Preprocessing...')

    # add in coordination set for regular training
    # reg_train_set = nid_train_set.concatenate(coord_set)

    #  FedAvg model
    if args.large_model:
        fedavg_model = tf.keras.Sequential()
        fedavg_model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(num_inputs,)))
        fedavg_model.add(tf.keras.layers.Dense(256, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(512, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(256, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(128, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(64, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(10, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))
    else:
        fedavg_model = tf.keras.Sequential()
        fedavg_model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(num_inputs,)))
        fedavg_model.add(tf.keras.layers.Dense(256, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(128, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(64, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(10, activation='relu'))
        fedavg_model.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))

    # Initialize Local Loss Function
    lossF = tf.keras.losses.SparseCategoricalCrossentropy()

    # model architecture
    layer_shapes, layer_sizes = get_model_architecture(fedavg_model)

    print(fedavg_model.summary())

    # initialize graph
    G = Graph(rank, size, mpi, args.graph_type, weight_type=args.weight_type)

    # initialize communicator
    communicator = DecentralizedSGD(rank, size, mpi, G, layer_shapes, layer_sizes)

    # L1, L2 penalties default to 1/2 a piece
    L1 = args.L1
    L2 = args.L2

    # epochs
    epochs = args.epochs

    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Output Path
    outputPath = 'Results/' + args.experiment
    saveFolder_fedavg = outputPath + '/' + args.name + '-' + str(size) + 'Worker-' + str(epochs) + 'Epochs-' + \
                        str(coordination_size) + 'Csize-' + str(args.graph_type)

    recorder_fedavg = Recorder(args.name, size, rank, args.graph_type, epochs, 0, coordination_size, outputPath,
                               save_folder_name=saveFolder_fedavg)

    mpi.Barrier()

    if rank == 0:
        with open(saveFolder_fedavg + '/ExpDescription', 'w') as f:
            f.write(str(args) + '\n')
        print('Beginning Training...')

    mpi.Barrier()

    # run MIDDLE training
    fedavg_train(fedavg_model, communicator, rank, lossF, optimizer, nid_train_set, coord_x, coord_y, nid_test_x,
                 nid_test_y, epochs, layer_shapes, layer_sizes, recorder_fedavg, L1, L2)

    # Plot confusion matrix
    fedavg_predictions = fedavg_model.predict(nid_test_x, verbose=0)

    # middle training
    pred = tf.math.argmax(fedavg_predictions, axis=1)
    fedavg_train_confusion_mtx = tf.math.confusion_matrix(nid_test_y, pred)
    # normalize confusion matrix
    fedavg_train_confusion_mtx = fedavg_train_confusion_mtx / tf.reduce_sum(fedavg_train_confusion_mtx, 0).numpy()
    fedavg_train_confusion_mtx = tf.where(tf.math.is_nan(fedavg_train_confusion_mtx),
                                          tf.zeros_like(fedavg_train_confusion_mtx), fedavg_train_confusion_mtx)

    # MIDDLE Training Results
    attack_labels = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    plt.figure(figsize=(8, 6))
    sns.heatmap(fedavg_train_confusion_mtx,
                xticklabels=attack_labels,
                yticklabels=attack_labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    fname = saveFolder_fedavg + '/r' + str(rank)
    tikzplotlib.save(fname + ".tex")
    # plt.title('FedAvg Confusion Matrix for Worker %d on CIC-Darknet2020 Data' % (rank + 1))
    plt.savefig(fname + '.pdf', format="pdf")
