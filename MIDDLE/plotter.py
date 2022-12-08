import numpy as np
import matplotlib.pyplot as plt
import os


def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data


def generate_confidence_interval(ys, number_per_g = 30, number_of_g = 1000, low_percentile = 1, high_percentile = 99):
    means = []
    mins =[]
    maxs = []
    for i,y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)


def plot_ci(x, y, num_runs, num_dots, mylegend,ls='-', lw=3, transparency=0.2):
    assert(x.ndim==1)
    assert(x.size==num_dots)
    assert(y.ndim==2)
    assert(y.shape==(num_runs,num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    plt.plot(x, y_mean, 'o-', label=mylegend, linestyle=ls, linewidth=lw) #, label=r'$\alpha$={}'.format(alpha))
    plt.fill_between(x, y_min, y_max, alpha=transparency)
    return


def unpack_data(directory_path, datatype='test-loss.log', epochs=10, num_workers=4):
    directory = os.path.join(directory_path)
    if not os.path.isdir(directory):
        raise Exception(f"custom no directory {directory}")
    data = np.zeros((epochs, num_workers))
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(datatype):
                j = int(file.split('-')[0][1:])
                with open(directory_path + '/' + file, 'r') as f:
                    i = 0
                    for line in f:
                        data[i, j] = line
                        i += 1
    return data


if __name__ == "__main__":

    workers = 4
    epochs = 10
    coordination_size = 128
    graph_type = 'ring'
    L1 = 1./3
    L3_vals = [0, 1. / 10, 1. / 8, 1. / 6, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    y_loss = []
    y_acc = []
    for trial in range(len(L3_vals)):
        L3 = L3_vals[trial]
        L2 = 1 - (L1 + L3)

        folder = 'Results/Darknet/MIDDLE' + '-' + str(workers) + 'Worker-' + str(epochs) + 'Epochs-' + \
                            str(L1) + 'L1Penalty-' + str(L2) + 'L2Penalty-' + str(coordination_size) + 'Csize-' + \
                            str(graph_type)

        test_loss_data = unpack_data(folder)
        test_acc_data = unpack_data(folder, datatype='test-acc.log')
        y_loss.append(test_loss_data.mean(axis=1))
        y_acc.append(test_acc_data.mean(axis=1))

        avg_worker_test_loss = np.mean(test_loss_data, axis=1)
        avg_worker_test_acc = np.mean(test_acc_data, axis=1)
        print('L3 Value: ' + str(L3))
        print('Test Loss')
        print(avg_worker_test_loss)
        print('Test Accuracy')
        print(avg_worker_test_acc)

    y_loss = np.stack(y_loss, axis=0)
    y_mean, y_min, y_max = generate_confidence_interval(y_loss)
    # print(y_min)
    # print(y_max)

    #plt.plot(range(1, epochs+1), y_mean)#, label=labels[ind % len(labels)] + exp_type[2], alpha=0.8, color=colors[ind])
    #plt.fill_between(range(1, epochs+1), y_min, y_max, alpha=0.2)#, color=colors[ind])
    #plt.show()
