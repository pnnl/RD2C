import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpi4py import MPI
import argparse
from MIDDLE_Train import middle_train, train
from dataloader import darknet_data
from communication import DecentralizedNoModelSGD
from misc import Recorder
from network import Graph
import os
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

    parser = argparse.ArgumentParser(description='MIDDLE Darknet Training')

    parser.add_argument('--name', '-n', default='MIDDLE', type=str, help='experiment name')
    parser.add_argument('--experiment', '-exp', default='Darknet', type=str, help='experiment name')
    parser.add_argument('--graph_type', default='ring', type=str, help='baseline topology')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='total epochs')
    parser.add_argument('--bs', default=64, type=int, help='train batch size for each worker')
    parser.add_argument('--train_split', default=0.8, type=float, help='train data percent')
    parser.add_argument('--coord_size', default=128, type=int, help='coordination dataset size')
    parser.add_argument('--L1', default=1/3, type=float, help='train set loss weighting')
    parser.add_argument('--L2', default=1/3, type=float, help='coordination set loss weighting')
    parser.add_argument('--L3', default=1/3, type=float, help='coordination set alignment loss weighting')
    parser.add_argument('--weight_type', default='uniform-symmetric', type=str, help='worker weightings')
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

    # initialize graph
    G = Graph(rank, size, mpi, args.graph_type, weight_type=args.weight_type, num_c=None)

    # initialize communicator
    communicator = DecentralizedNoModelSGD(rank, size, mpi, G)

    #  MIDDLE model
    if rank == 0 or rank == 1:
        middle_model = tf.keras.Sequential()
        middle_model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(num_inputs,)))
        middle_model.add(tf.keras.layers.Dense(256, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(512, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(256, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(128, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(64, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(10, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))
    else:
        middle_model = tf.keras.Sequential()
        middle_model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(num_inputs,)))
        middle_model.add(tf.keras.layers.Dense(256, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(128, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(64, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(10, activation='relu'))
        middle_model.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))

    # Initialize Local Loss Function
    lossF = tf.keras.losses.SparseCategoricalCrossentropy()

    # model architecture
    layer_shapes, layer_sizes = get_model_architecture(middle_model)

    # L1, L2, L3 penalties
    L1 = args.L1
    L3 = args.L3
    L2 = 1 - (L1 + L3)
    if rank == 0:
        print('L3 Value = %f' % L3)

    # epochs
    epochs = args.epochs

    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Create MIDDLE model (same architecture and weights) for comparison
    # middle_model_initial = tf.keras.models.clone_model(middle_model)

    # Output Path
    outputPath = 'Results/' + args.experiment
    saveFolder_middle = outputPath + '/' + args.name + '-' + str(size) + 'Worker-' + str(epochs) + 'Epochs-' + \
                        str(L1) + 'L1Penalty-' + str(L2) + 'L2Penalty-' + str(coordination_size) + 'Csize-' + \
                        str(args.graph_type)

    recorder_middle = Recorder(args.name, size, rank, args.graph_type, epochs, L1, L2, coordination_size, outputPath)

    mpi.Barrier()

    if rank == 0:
        with open(saveFolder_middle + '/ExpDescription', 'w') as f:
            f.write(str(args) + '\n')
        print('Beginning Training...')

    mpi.Barrier()

    # run MIDDLE training
    middle_train(middle_model, communicator, rank, lossF, optimizer, nid_train_set, coord_x, coord_y, nid_test_x,
                 nid_test_y, epochs, coordination_size, num_outputs, layer_shapes, layer_sizes, recorder_middle,
                 L1, L2, L3)

    # Plot confusion matrix
    middle_predictions = middle_model.predict(nid_test_x, verbose=0)

    # middle training
    pred_middle = tf.math.argmax(middle_predictions, axis=1)
    middle_train_confusion_mtx = tf.math.confusion_matrix(nid_test_y, pred_middle)
    # normalize confusion matrix
    middle_train_confusion_mtx = middle_train_confusion_mtx / tf.reduce_sum(middle_train_confusion_mtx, 0).numpy()
    middle_train_confusion_mtx = tf.where(tf.math.is_nan(middle_train_confusion_mtx),
                                          tf.zeros_like(middle_train_confusion_mtx), middle_train_confusion_mtx)

    # MIDDLE Training Results
    attack_labels = ['Non-Tor', 'NonVPN', 'Tor', 'VPN']
    plt.figure(figsize=(8, 6))
    sns.heatmap(middle_train_confusion_mtx,
                xticklabels=attack_labels,
                yticklabels=attack_labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('True Label')
    plt.title('MIDDLE Confusion Matrix for Worker %d on CIC-Darknet2020 Data' % (rank + 1))
    plt.savefig(saveFolder_middle + '/Regular-r' + str(rank) + '.pdf', format="pdf")
