import numpy as np
import matplotlib.pyplot as plt
import os
import tikzplotlib
# plt.rcParams['text.usetex'] = True
plt.rcParams["mathtext.fontset"]


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

    test_loss = False
    workers = 4
    epochs = 50
    coordination_size = 128
    graph_type = 'ring'
    resultFolder = 'Results/Darknet/' + str(workers) + 'WorkerRing/MIDDLE-'

    #graph_type = 'fully-connected'
    #resultFolder = 'Results/Darknet/' + str(workers) + 'WorkerFC/MIDDLE-'

    #graph_type = 'clique-ring'
    #resultFolder = 'Results/Darknet/' + str(workers) + 'WorkerROC/MIDDLE-'

    L1 = 1./3
    # L3_vals = [0, 1. / 20, 1. / 10, 1./6, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    # L3_vals = [0.0, 1. / 20, 1. / 10, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    L3_vals = [0.0, 1. / 10, 1. / 4, 1. / 3, 1. / 2, 3. / 5]
    legend_vals = ['0', '1/10', '1/4', '1/3', '1/2', '3/5']
    # L3_vals = [0.0, 1. / 20, 1. / 4, 1. / 3, 1. / 2]
    runs = 6
    plt.figure(1)

    mean_accs = np.empty(len(legend_vals))
    mean_losses = np.empty(len(legend_vals))

    for ind, val in enumerate(range(len(L3_vals))):
        y_loss = []
        y_acc = []
        L3 = L3_vals[val]

        mean_acc = np.zeros(epochs)
        mean_loss = np.zeros(epochs)

        for run in range(1, runs):
            folder = resultFolder + str(run) + '-' + str(workers) + 'Worker-' + str(epochs) + 'Epochs-' + \
                                str(L3) + 'L3Penalty-' + str(coordination_size) + 'Csize-' + str(graph_type)

            test_loss_data = unpack_data(folder, epochs=epochs, num_workers=workers)
            test_acc_data = unpack_data(folder, datatype='test-acc.log', epochs=epochs, num_workers=workers)
            y_loss.append(test_loss_data.mean(axis=1))
            y_acc.append(test_acc_data.mean(axis=1))

            avg_worker_test_loss = np.mean(test_loss_data, axis=1)
            avg_worker_test_acc = np.mean(test_acc_data, axis=1)

            mean_acc += avg_worker_test_acc
            mean_loss += avg_worker_test_loss

            if run == runs-1:
                print('L3 Value: ' + str(L3))
                print('Mean Test Loss')
                print(mean_loss[-1]/(runs-1))
                mean_losses[ind] = mean_loss[-1]/(runs-1)
                print('Mean Test Accuracy')
                print(mean_acc[-1]/(runs-1))
                mean_accs[ind] = mean_acc[-1]/(runs-1)


        '''
        y_loss = np.stack(y_loss, axis=0)
        y_acc = np.stack(y_acc, axis=0)
        y_mean, y_min, y_max = generate_confidence_interval(y_loss)
        y_mean_a, y_min_a, y_max_a = generate_confidence_interval(y_acc)
        # mylegend = r'$\lambda_3 =$' + str(L3)
        mylegend = r'$\lambda_3 = $' + legend_vals[ind]
        if test_loss:
            plt.plot(range(1, epochs+1), y_mean, label=mylegend, alpha=0.8) #, color=colors[ind])
            plt.fill_between(range(1, epochs+1), y_min, y_max, alpha=0.2)#, color=colors[ind])
        else:
            plt.plot(range(1, epochs + 1), y_mean_a, label=mylegend, alpha=0.8)  # , color=colors[ind])
            plt.fill_between(range(1, epochs + 1), y_min_a, y_max_a, alpha=0.2)  # , color=colors[ind])

    if test_loss:
        plt.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Average Test Loss', fontsize=15)
        plt.grid()
        plt.xlim([1, 50])
        #fig = plt.gcf()
        #fig.set_size_inches(18.5, 18.5)
        saveFilename = "test-loss-" + str(workers) + graph_type + ".pdf"
    else:
        plt.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Average Test Accuracy', fontsize=15)
        plt.grid()
        plt.xlim([1, 50])
        plt.ylim([0.845, 0.87])
        # plt.ylim([0.835, 0.885])
        #fig = plt.gcf()
        #fig.set_size_inches(18.5, 10.5)
        saveFilename = "test-acc-" + str(workers) + graph_type + ".pdf"

    plt.savefig(saveFilename, format="pdf")
    # tikzplotlib.save(saveFilename[:-4] + ".tex")
    plt.show()
    '''

    #'''
    X = [r'$\lambda_3 = 0$', r'$\lambda_3 = 1/10$', r'$\lambda_3 = 1/4$', r'$\lambda_3 = 1/3$', r'$\lambda_3 = 1/2$',
         r'$\lambda_3 = 3/5$']

    X_axis = np.arange(len(X))
    print(mean_accs)

    graph = plt.bar(X_axis - 0.1, mean_accs*100, 0.5, label='Average Test Accuracy', color='skyblue')
    graph2 = plt.bar(X_axis + 0.1, mean_losses, 0.5, label='Average Test Loss', color='khaki')

    i = 0
    for p in graph2:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height * 1.01,
                 str(round(mean_losses[i], 2)),
                 ha='center',
                 weight='bold',
                 fontsize=10)
        i += 1

    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height * 1.01,
                 str(round(mean_accs[i]*100, 1)) + '%',
                 ha='center',
                 weight='bold',
                 fontsize=10)
        i += 1

    plt.xticks(X_axis, X)
    plt.ylim(0, 100)
    plt.xlabel(r"Varying Collaboration Weighting $\lambda_3$", fontsize=15)
    plt.ylabel("Final Average Test Accuracy (%) / Loss", fontsize=15)
    plt.legend(loc='best', ncol=2, fancybox=True, framealpha=0)
    saveFilename = "varyingL3Bar.pdf"
    plt.show()
    #plt.savefig(saveFilename, format="pdf")
    #'''

