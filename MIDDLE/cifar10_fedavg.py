import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import os
import platform
from mpi4py import MPI
from FedAvg_Train import fedavg_train
from communication import DecentralizedSGD
from misc import Recorder
from network import Graph
import argparse
from models.resnet import ResNet18
tf.config.set_visible_devices([], 'GPU')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 32, 32, 3)
        #X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(rank, size, num_validation, train_bs, num_test=10000, skew_factor=0.75):

    num_training = 50000 - num_validation

    # Load the raw CIFAR-10 data
    cifar10_dir = 'Data/cifar10/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # skew data here...
    skew_sort_idx = np.argsort(y_train)
    skew_idx = skew_sort_idx[:int(num_training*skew_factor)]
    non_skew_idx = skew_sort_idx[int(num_training*skew_factor):]
    np.random.shuffle(non_skew_idx)

    # split skewed data amongst devices
    x_train_s = np.array_split(x_train[skew_idx, :, :, :], size)[rank]

    y_train_s = np.array_split(y_train[skew_idx], size)[rank]

    # split random data amongst devices
    x_train_r = np.array_split(x_train[non_skew_idx, :, :, :], size)[rank]
    y_train_r = np.array_split(y_train[non_skew_idx], size)[rank]

    # combine skew and non-skewed data
    x_train = np.concatenate((x_train_s, x_train_r), axis=0)
    y_train = np.hstack((y_train_s, y_train_r))

    # create TF datasets
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(int(num_training/size)).batch(train_bs)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    return train_data, x_test, y_test, tf.convert_to_tensor(X_val), tf.convert_to_tensor(y_val), x_train, y_train


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

    parser = argparse.ArgumentParser(description='FedAvg CIFAR10 Training')
    parser.add_argument('--name', '-n', default='FedAvg', type=str, help='experiment name')
    parser.add_argument('--experiment', '-exp', default='Cifar10', type=str, help='experiment type')
    parser.add_argument('--skew', '-s', default=0.75, type=float, help='skew factor')
    parser.add_argument('--graph_type', default='ring', type=str, help='baseline topology')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', default=20, type=int, help='total epochs')
    parser.add_argument('--bs', default=256, type=int, help='train batch size for each worker')
    parser.add_argument('--train_split', default=0.8, type=float, help='train data percent')
    parser.add_argument('--coord_size', default=256, type=int, help='coordination dataset size')
    parser.add_argument('--L1', default=1., type=float, help='train set loss weighting')
    parser.add_argument('--L2', default=1., type=float, help='coordination set loss weighting')
    parser.add_argument('--L3', default=1., type=float, help='coordination set alignment loss weighting')
    parser.add_argument('--weight_type', default='uniform-neighbor-no-self-weight', type=str, help='worker weightings')
    parser.add_argument('--randomSeed', default=482, type=int, help='random seed')
    parser.add_argument('--large_model', default=1, type=int, help='use larger model for fedavg')
    args = parser.parse_args()

    # initialize random seed
    tf.keras.utils.set_random_seed(args.randomSeed)

    mpi = MPI.COMM_WORLD
    bcast = mpi.bcast
    barrier = mpi.barrier
    rank = mpi.rank
    size = mpi.size

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_outputs = len(classes)

    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)

    # Invoke the above function to get our data.
    train_data, x_test, y_test, x_val, y_val, x_train, y_train = get_CIFAR10_data(rank, size, args.coord_size, args.bs,
                                                                                  skew_factor=args.skew)

    #  FedAvg model
    if args.large_model:
        model = ResNet18(10)
        model.build(input_shape=(None, 32, 32, 3))
        lr = args.lr
    else:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        lr = 0.01

    # initialize graph
    G = Graph(rank, size, mpi, args.graph_type, weight_type=args.weight_type)

    # model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    # initialize communicator
    communicator = DecentralizedSGD(rank, size, mpi, G, layer_shapes, layer_sizes)

    # Initialize Local Loss Function
    lossF = tf.keras.losses.SparseCategoricalCrossentropy()

    # L1, L2, L3 penalties
    L1 = args.L1
    L2 = args.L2

    sum = L1+L2
    L1 = L1 / sum
    L2 = L2 / sum

    # epochs
    epochs = args.epochs

    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Output Path
    outputPath = 'Results/' + args.experiment
    saveFolder_fedavg = outputPath + '/' + args.name + '-' + str(size) + 'Worker-' + str(epochs) + 'Epochs-' +\
                        str(args.coord_size) + 'Csize-' + str(args.skew) + 'Skew-' + str(args.graph_type)

    recorder_fedavg = Recorder(args.name, size, rank, args.graph_type, epochs, 0, args.coord_size, args.skew,
                               outputPath, save_folder_name=saveFolder_fedavg)

    mpi.Barrier()

    if rank == 0:
        with open(saveFolder_fedavg + '/ExpDescription', 'w') as f:
            f.write(str(args) + '\n')
        print('Beginning Training...')

    mpi.Barrier()

    fedavg_train(model, communicator, rank, lossF, optimizer, train_data, x_val, y_val, x_test,
                 y_test, epochs, layer_shapes, layer_sizes, recorder_fedavg, L1, L2)
