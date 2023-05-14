import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import os
import platform
from mpi4py import MPI
from MIDDLE_Train import middle_train
from communication import DecentralizedNoModelSGD
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

    parser = argparse.ArgumentParser(description='MIDDLE CIFAR10 Training')
    parser.add_argument('--name', '-n', default='MIDDLE-CIFAR10', type=str, help='experiment name')
    parser.add_argument('--experiment', '-exp', default='Cifar10', type=str, help='experiment type')
    parser.add_argument('--skew', '-s', default=0.75, type=float, help='skew factor')
    parser.add_argument('--graph_type', default='fully-connected', type=str, help='baseline topology')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', default=20, type=int, help='total epochs')
    parser.add_argument('--bs', default=256, type=int, help='train batch size for each worker')
    parser.add_argument('--multi_model', default=1, type=int, help='workers use different models during training')
    parser.add_argument('--coord_size', default=256, type=int, help='coordination dataset size')
    parser.add_argument('--L1', default=1., type=float, help='train set loss weighting')
    parser.add_argument('--L2', default=1., type=float, help='coordination set loss weighting')
    parser.add_argument('--L3', default=1., type=float, help='coordination set alignment loss weighting')
    parser.add_argument('--weight_type', default='uniform-neighbor-no-self-weight', type=str, help='worker weightings')
    parser.add_argument('--randomSeed', default=482, type=int, help='random seed')
    parser.add_argument('--large_model', default=1, type=int, help='use larger model')
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

    if args.multi_model:
        # create model
        if rank < int(size/2):
            # use a CNN for first half of workers
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

        else:
            model = ResNet18(10)
            model.build(input_shape=(None, 32, 32, 3))
            lr = args.lr
    else:
        if args.large_model:
            # Resnet 18
            model = ResNet18(10)
            model.build(input_shape=(None, 32, 32, 3))
            lr = args.lr

            '''
            # ResNet50
            model = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                                            weights=None,
                                                            input_tensor=None,
                                                            input_shape=(32, 32, 3),
                                                            pooling=None,
                                                            classes=10,
                                                            )
            '''
        else:
            # small model
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

    # initialize communicator
    communicator = DecentralizedNoModelSGD(rank, size, mpi, G)

    # Initialize Local Loss Function
    lossF = tf.keras.losses.SparseCategoricalCrossentropy()

    # model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    # L1, L2, L3 penalties
    L1 = args.L1
    L2 = args.L2
    L3 = args.L3

    sum = L1+L2+L3
    L1 = L1 / sum
    L2 = L2 / sum
    L3 = L3 / sum

    '''
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=256,
              epochs=10,
              validation_data=(x_test, y_test),
              shuffle=True)
    '''

    if rank == 0:
        print('L3 Value = %f' % L3)

    # epochs
    epochs = args.epochs

    # Initialize Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Output Path
    outputPath = 'Results/' + args.experiment
    saveFolder_middle = outputPath + '/' + args.name + '-' + str(size) + 'Worker-' + str(epochs) + 'Epochs-' + \
                        str(L3) + 'L3Penalty-' + str(args.coord_size) + 'Csize-' + str(args.skew) + 'Skew-' \
                        + str(args.graph_type)

    recorder_middle = Recorder(args.name, size, rank, args.graph_type, epochs, L3, args.coord_size, args.skew,
                               outputPath)

    mpi.Barrier()

    if rank == 0:
        with open(saveFolder_middle + '/ExpDescription', 'w') as f:
            f.write(str(args) + '\n')
        print('Beginning Training...')

    mpi.Barrier()

    middle_train(model, communicator, rank, lossF, optimizer, train_data, x_val, y_val, x_test,
                 y_test, epochs, args.coord_size, num_outputs, layer_shapes, layer_sizes, recorder_middle,
                 L1, L2, L3)
