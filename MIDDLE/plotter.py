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


def process_data(workers, epochs=50, coordination_size=128, graph_type='ring', resultFolder='Results/Darknet/',
                 method='middle', runs=6):

    if method == 'middle':
        if graph_type == 'ring':
            resultFolder = resultFolder + str(workers) + 'WorkerRing/MIDDLE-'
        elif graph_type == 'fully-connected':
            resultFolder = resultFolder + str(workers) + 'WorkerFC/MIDDLE-'
        elif graph_type == 'clique-ring':
            resultFolder = resultFolder + str(workers) + 'WorkerROC/MIDDLE-'
    elif method == 'fedavg-large':
        if graph_type == 'ring':
            resultFolder = resultFolder + 'FedAvgLarge' + str(workers) + 'WorkerRing/FedAvg-'
        elif graph_type == 'fully-connected':
            resultFolder = resultFolder + 'FedAvgLarge' + str(workers) + 'WorkerFC/FedAvg-'
        elif graph_type == 'clique-ring':
            resultFolder = resultFolder + 'FedAvgLarge' + str(workers) + 'WorkerROC/FedAvg-'
    elif method == 'fedavg-small':
        if graph_type == 'ring':
            resultFolder = resultFolder + 'FedAvgSmall' + str(workers) + 'WorkerRing/FedAvg-'
        elif graph_type == 'fully-connected':
            resultFolder = resultFolder + 'FedAvgSmall' + str(workers) + 'WorkerFC/FedAvg-'
        elif graph_type == 'clique-ring':
            resultFolder = resultFolder + 'FedAvgSmall' + str(workers) + 'WorkerROC/FedAvg-'

    # L3_vals = [0, 1. / 20, 1. / 10, 1./6, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    # L3_vals = [0.0, 1. / 20, 1. / 10, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    L3_vals = [0.0, 1. / 10, 1. / 4, 1. / 3, 1. / 2, 3. / 5]
    plt.figure(1)
    mean_l = []; min_l = []; max_l = []
    mean_a = []; min_a = []; max_a = []

    for ind, val in enumerate(range(len(L3_vals))):
        y_loss = []
        y_acc = []
        L3 = L3_vals[val]

        for run in range(1, runs):
            folder = resultFolder + str(run) + '-' + str(workers) + 'Worker-' + str(epochs) + 'Epochs-' + \
                     str(L3) + 'L3Penalty-' + str(coordination_size) + 'Csize-' + str(graph_type)

            test_loss_data = unpack_data(folder, epochs=epochs, num_workers=workers)
            test_acc_data = unpack_data(folder, datatype='test-acc.log', epochs=epochs, num_workers=workers)
            y_loss.append(test_loss_data.mean(axis=1))
            y_acc.append(test_acc_data.mean(axis=1))

        y_loss = np.stack(y_loss, axis=0)
        y_acc = np.stack(y_acc, axis=0)
        y_mean_l, y_min_l, y_max_l = generate_confidence_interval(y_loss)
        y_mean_a, y_min_a, y_max_a = generate_confidence_interval(y_acc)
        mean_l.append(y_mean_l); min_l.append(y_min_l); max_l.append(y_max_l)
        mean_a.append(y_mean_a); min_a.append(y_min_a); max_a.append(y_max_a)

    return mean_l, min_l, max_l, mean_a, min_a, max_a, len(L3_vals)


def results_bar_chart(mean_accs, mean_losses, save_fig=True):

    m_len = len(mean_accs)
    final_mean_accs = np.empty(m_len)
    final_mean_losses = np.empty(m_len)
    for i in range(m_len):
        final_mean_accs[i] = mean_accs[i][-1]
        final_mean_losses[i] = mean_losses[i][-1]

    X = [r'$\lambda_3 = 0$', r'$\lambda_3 = 1/10$', r'$\lambda_3 = 1/4$', r'$\lambda_3 = 1/3$', r'$\lambda_3 = 1/2$',
         r'$\lambda_3 = 3/5$']

    X_axis = np.arange(len(X))

    graph = plt.bar(X_axis - 0.1, final_mean_accs * 100, 0.5, label='Average Test Accuracy', color='skyblue')
    graph2 = plt.bar(X_axis + 0.1, final_mean_losses, 0.5, label='Average Test Loss', color='khaki')

    i = 0
    for p in graph2:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x + width / 2,
                 y + height * 1.01,
                 str(round(final_mean_losses[i], 2)),
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
                 str(round(final_mean_accs[i] * 100, 1)) + '%',
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
    if save_fig:
        plt.savefig(saveFilename, format="pdf")
        # tikzplotlib.save(saveFilename[:-4] + ".tex")
    else:
        plt.show()


def plot_acc_loss(mean_l, min_l, max_l, mean_a, min_a, max_a, lenL3, epochs=50, plot_loss=True, save_fig=True):
    legend_vals = ['0', '1/10', '1/4', '1/3', '1/2', '3/5']
    for i in range(lenL3):
        mylegend = r'$\lambda_3 = $' + legend_vals[i]
        if plot_loss:
            y_mean = mean_l[i]
            y_min = min_l[i]
            y_max = max_l[i]
            plt.plot(range(1, epochs + 1), y_mean, label=mylegend, alpha=0.8)  # , color=colors[ind])
            plt.fill_between(range(1, epochs + 1), y_min, y_max, alpha=0.2)  # , color=colors[ind])
        else:
            y_mean = mean_a[i]
            y_min = min_a[i]
            y_max = max_a[i]
            plt.plot(range(1, epochs + 1), y_mean, label=mylegend, alpha=0.8)  # , color=colors[ind])
            plt.fill_between(range(1, epochs + 1), y_min, y_max, alpha=0.2)  # , color=colors[ind])

    if plot_loss:
        plt.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Average Test Loss', fontsize=15)
        plt.grid()
        plt.xlim([1, 50])
        saveFilename = "test-loss-" + str(workers) + graph_type + ".pdf"
    else:
        plt.legend(loc='upper left', ncol=2, fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Average Test Accuracy', fontsize=15)
        plt.grid()
        plt.xlim([1, 50])
        plt.ylim([0.845, 0.87])
        # plt.ylim([0.835, 0.885])
        saveFilename = "test-acc-" + str(workers) + graph_type + ".pdf"

    if save_fig:
        plt.savefig(saveFilename, format="pdf")
        # tikzplotlib.save(saveFilename[:-4] + ".tex")
    else:
        plt.show()


if __name__ == "__main__":

    plot_loss = True
    workers = 4
    graph_type = 'ring'
    resultFolder = 'Results/Darknet/'

    mean_l, min_l, max_l, mean_a, min_a, max_a,  lenL3 = process_data(workers, graph_type=graph_type)
    plot_acc_loss(mean_l, min_l, max_l, mean_a, min_a, max_a, lenL3, plot_loss=plot_loss, save_fig=False)
    results_bar_chart(mean_a, mean_l, save_fig=False)
