import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpi4py import MPI
import argparse
from MIDDLE_Train import middle_train, train
from communication import DecentralizedNoModelSGD
from misc import Recorder
from network import Graph


def get_model_architecture(model):
    # find shape and total elements for each layer of the resnet model
    model_weights = model.get_weights()
    layer_shapes = []
    layer_sizes = []
    for i in range(len(model_weights)):
        layer_shapes.append(model_weights[i].shape)
        layer_sizes.append(model_weights[i].size)
    return layer_shapes, layer_sizes


def data_pre_process(rank, size, train_pct, train_bs, test_bs, coordination_size):
    # read in CSV data
    raw_df_data = pd.read_csv("Experiments/Darknet2020/Data/Darknet.CSV", parse_dates=["Timestamp"], on_bad_lines='skip')

    # make timestamp numeric (just time of day)
    timestamp = raw_df_data["Timestamp"]
    raw_df_data["Timestamp"] = timestamp.dt.hour + timestamp.dt.minute / 60 + timestamp.dt.second / 3600

    # remove NaN rows and Inf
    raw_df = raw_df_data.replace([np.inf, -np.inf], np.nan)
    raw_df.dropna(inplace=True)

    # clean up the sub-labels (incorrectly labeled)
    raw_df.loc[raw_df['Label.1'] == 'AUDIO-STREAMING', 'Label.1'] = 'Audio-Streaming'
    raw_df.loc[raw_df['Label.1'] == 'AUDIO-STREAMING', 'Label.1'] = 'Audio-Streaming'
    raw_df.loc[raw_df['Label.1'] == 'Video-streaming', 'Label.1'] = 'Video-Streaming'
    raw_df.rename(columns={"Label.1": "Subtype"}, inplace=True)

    # add one-hot encoding of the sub-labels
    onehot = pd.get_dummies(raw_df['Subtype'])
    raw_df = pd.concat([raw_df, onehot], axis=1, join='inner')

    # drop IP columns and 0 columns
    ip_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Active Mean', 'Active Std', 'Active Max',
               'Active Min', 'Subflow Bwd Packets', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
               'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'URG Flag Count', 'CWE Flag Count',
               'ECE Flag Count', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags']
    raw_df.drop(ip_cols, axis=1, inplace=True)

    # label dataframe
    traffic_categories = raw_df['Label'].unique()
    tc = dict(zip(traffic_categories, range(len(traffic_categories))))
    class_attack = raw_df.Label.map(lambda a: tc[a])
    raw_df['Label'] = class_attack

    # shuffle dataset
    raw_df = raw_df.sample(frac=1)

    # extract features
    non_normalized_df = raw_df.drop(['Label', 'Subtype'], axis=1)

    # extract labels
    labels = raw_df['Label']

    # normalize the feature dataframe
    normalized_df = non_normalized_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # create coordination set
    coord_x = tf.convert_to_tensor(normalized_df.iloc[:coordination_size, :])
    coord_y = tf.convert_to_tensor(labels[:coordination_size])
    coordination_set = tf.data.Dataset.from_tensor_slices((coord_x, coord_y)).batch(coordination_size)

    # get data info
    num_inputs = len(normalized_df.columns.to_list())
    num_outputs = len(traffic_categories)

    # Split training data amongst workers
    worker_data = np.array_split(normalized_df.iloc[coordination_size:, :], size)[rank]
    worker_label = np.array_split(labels[coordination_size:], size)[rank]

    # create train/test split
    num_data = len(worker_label)
    num_train = int(num_data * train_pct)
    # train
    train_x = tf.convert_to_tensor(worker_data.iloc[:num_train, :])
    train_y = tf.convert_to_tensor(worker_label[:num_train])
    train_set = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(train_bs)
    # test
    test_x = tf.convert_to_tensor(worker_data.iloc[num_train:, :])
    test_y = tf.convert_to_tensor(worker_label[num_train:])
    test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(test_bs)

    # non-iid Dataset
    # create train/test split
    nid_num_data = len(labels[coordination_size:])
    nid_num_train = int(nid_num_data * train_pct)
    normalized_df['Label'] = labels

    # Skew Non-IID
    nid_test_df = normalized_df[(coordination_size + nid_num_train):]
    test_labels = nid_test_df['Label']
    skew_df_sort = normalized_df[coordination_size:(coordination_size + nid_num_train)].sort_values(by=['Label'])
    nid_labels = skew_df_sort['Label']
    skew_df_sort = skew_df_sort.drop(['Label'], axis=1)
    nid_test_df = nid_test_df.drop(['Label'], axis=1)
    nid_worker_data = np.array_split(skew_df_sort, size)[rank]
    nid_worker_label = np.array_split(nid_labels, size)[rank]

    # Non-IID train
    nid_train_x = tf.convert_to_tensor(nid_worker_data)
    nid_train_y = tf.convert_to_tensor(nid_worker_label)
    nid_train_set = tf.data.Dataset.from_tensor_slices((nid_train_x, nid_train_y)).batch(train_bs)
    # Non-IID test
    nid_test_x = tf.convert_to_tensor(nid_test_df)
    nid_test_y = tf.convert_to_tensor(test_labels)

    return train_set, test_set, coordination_set, nid_train_set, nid_test_x, nid_test_y, num_inputs, num_outputs


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
    train_set =None; test_set = None; coord_set = None; nid_train_set = None; nid_test_x = None; nid_test_y = None
    num_inputs = None; num_outputs = None

    if rank == 0:
        print('Beginning Data Preprocessing...')

    if args.experiment == 'Darknet':
        train_set, test_set, coord_set, nid_train_set, nid_test_x, nid_test_y, num_inputs, num_outputs = \
            data_pre_process(rank, size, train_pct, train_bs, test_bs, coordination_size)

    if rank == 0:
        print('Finished Data Preprocessing...')

    # add in coordination set for regular training
    reg_train_set = nid_train_set.concatenate(coord_set)

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

    # epochs
    epochs = args.epochs

    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Create MIDDLE model (same architecture and weights) for comparison
    initial_weights = middle_model.get_weights()
    # middle_model_initial = tf.keras.models.clone_model(middle_model)

    L1 = 1./3
    # L3_vals = [0, 1. / 10, 1. / 8, 1. / 6, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    L3_vals = [1. / 6, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3, 0, 1. / 10, 1. / 8]
    runs = 5

    # run MIDDLE ablation
    for run in range(1, runs+1):

        # initialize random seed
        np.random.seed(args.randomSeed + run - 1)
        tf.random.set_seed(args.randomSeed + run - 1)
        for trial in range(len(L3_vals)):

            # load initial weights
            middle_model.set_weights(initial_weights)

            L3 = L3_vals[trial]
            L2 = 1 - (L1 + L3)
            if rank == 0:
                print('L3 Value = %f' % L3)

            # Output Path
            outputPath = 'Results/' + args.experiment
            name = args.name + str(run)
            saveFolder_middle = outputPath + '/' + name + '-' + str(size) + 'Worker-' + str(epochs) + 'Epochs-' + \
                                str(L1) + 'L1Penalty-' + str(L2) + 'L2Penalty-' + str(coordination_size) + 'Csize-' + \
                                str(args.graph_type)

            recorder_middle = Recorder(name, size, rank, args.graph_type, epochs, L1, L2, coordination_size, outputPath)

            mpi.Barrier()

            if rank == 0:
                with open(saveFolder_middle + '/ExpDescription', 'w') as f:
                    f.write(str(args) + '\n')
                    f.write('L1 = ' + str(L1) + '\n')
                    f.write('L2 = ' + str(L2) + '\n')
                    f.write('L3 = ' + str(L3) + '\n')
                    f.write('Seed = ' + str(run) + '\n')
                print('Beginning Training...')

            mpi.Barrier()

            middle_train(middle_model, communicator, rank, lossF, optimizer, nid_train_set, coord_set, nid_test_x,
                         nid_test_y, epochs, coordination_size, num_outputs, layer_shapes, layer_sizes, recorder_middle,
                         L1, L2, L3)

            mpi.Barrier()

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

            mpi.Barrier()

            # middle_model = tf.keras.models.clone_model(middle_model_initial)
