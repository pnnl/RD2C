import numpy as np
import matplotlib.pyplot as plt
import os
import tikzplotlib
# plt.rcParams['text.usetex'] = True
plt.rcParams["mathtext.fontset"]


def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw))
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw))
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)

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
        L3_vals = [0.0, 1. / 10, 1. / 4, 1. / 3, 1. / 2, 3. / 5]
        if graph_type == 'ring':
            resultFolder = resultFolder + str(workers) + 'WorkerRing/MIDDLE-'
        elif graph_type == 'fully-connected':
            resultFolder = resultFolder + str(workers) + 'WorkerFC/MIDDLE-'
        elif graph_type == 'clique-ring':
            resultFolder = resultFolder + str(workers) + 'WorkerROC/MIDDLE-'
    elif method == 'fedavg-large':
        L3_vals = [0.0]
        if graph_type == 'ring':
            resultFolder = resultFolder + 'FedAvgLarge/' + str(workers) + 'WorkerRing/FedAvg-'
        elif graph_type == 'fully-connected':
            resultFolder = resultFolder + 'FedAvgLarge/' + str(workers) + 'WorkerFC/FedAvg-'
        elif graph_type == 'clique-ring':
            resultFolder = resultFolder + 'FedAvgLarge/' + str(workers) + 'WorkerROC/FedAvg-'
    elif method == 'fedavg-small':
        L3_vals = [0.0]
        if graph_type == 'ring':
            resultFolder = resultFolder + 'FedAvgSmall/' + str(workers) + 'WorkerRing/FedAvg-'
        elif graph_type == 'fully-connected':
            resultFolder = resultFolder + 'FedAvgSmall/' + str(workers) + 'WorkerFC/FedAvg-'
        elif graph_type == 'clique-ring':
            resultFolder = resultFolder + 'FedAvgSmall/' + str(workers) + 'WorkerROC/FedAvg-'

    # L3_vals = [0, 1. / 20, 1. / 10, 1./6, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    # L3_vals = [0.0, 1. / 20, 1. / 10, 1. / 4, 1. / 3, 1. / 2, 3. / 5, 2. / 3]
    plt.figure(1)
    mean_l = []; min_l = []; max_l = []
    mean_a = []; min_a = []; max_a = []

    for ind, val in enumerate(range(len(L3_vals))):
        y_loss = []
        y_acc = []
        L3 = L3_vals[val]

        for run in range(1, runs):
            if method == 'middle':
                folder = resultFolder + str(run) + '-' + str(workers) + 'Worker-' + str(epochs) + 'Epochs-' + \
                        str(L3) + 'L3Penalty-' + str(coordination_size) + 'Csize-' + str(graph_type)
            elif method[:3] == 'fed':
                folder = resultFolder + str(run) + '-' + str(workers) + 'Worker-' + str(epochs) + 'Epochs-' + \
                         str(coordination_size) + 'Csize-' + str(graph_type)

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
        # plt.ylim([0.845, 0.87])
        # plt.ylim([0.835, 0.885])
        saveFilename = "test-acc-" + str(workers) + graph_type + ".pdf"

    if save_fig:
        plt.savefig(saveFilename, format="pdf")
        # tikzplotlib.save(saveFilename[:-4] + ".tex")
    else:
        plt.show()


if __name__ == "__main__":

    plot_loss = False
    workers = 16
    graph_type = 'ring'
    resultFolder = 'Results/Darknet/'
    method = 'middle'

    mean_l, min_l, max_l, mean_a, min_a, max_a,  lenL3 = process_data(workers, graph_type=graph_type, method=method)
    plot_acc_loss(mean_l, min_l, max_l, mean_a, min_a, max_a, lenL3, plot_loss=plot_loss, save_fig=False)
    if method == 'middle':
        results_bar_chart(mean_a, mean_l, save_fig=False)

    '''
    # FedAvg Bar Chart
    mean_l, _, _, mean_a, _, _, _ = process_data(16, graph_type='fully-connected', method='fedavg-large')
    middle_mean_l, _, _, middle_mean_a, _, _, _ = process_data(16, graph_type='fully-connected', method='middle')
    mean_l = mean_l[-1][-1]
    mean_a = mean_a[-1][-1]
    middle_nol3_l = middle_mean_l[0][-1]
    middle_nol3_a = middle_mean_a[0][-1]
    middle_l3_1L = middle_mean_l[1][-1]
    middle_l3_1A = middle_mean_a[1][-1]
    middle_l3_2L = middle_mean_l[2][-1]
    middle_l3_2A = middle_mean_a[2][-1]
    middle_l3_3L = middle_mean_l[3][-1]
    middle_l3_3A = middle_mean_a[3][-1]
    save_fig = False

    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx()

    accs = np.array([mean_a, middle_nol3_a, middle_l3_1A, middle_l3_2A, middle_l3_3A])
    losses = np.array([mean_l, middle_nol3_l, middle_l3_1L, middle_l3_2L, middle_l3_3L])
    params = np.array([347382, 0, 512, 512, 512])
    # log_params = np.where(params > 0, np.log(params), 0)

    X = ['FedAvg', 'No Collaboration', r'MIDDLE $\lambda_3 = 1/10$', r'MIDDLE $\lambda_3 = 1/4$', r'MIDDLE $\lambda_3 = 1/3$']
    X_axis = np.arange(len(X))

    ax.bar(np.nan, np.nan, 0.5, label='Total Communicated \n Parameters', color='lightgreen')
    graph = ax.bar(X_axis, accs * 100, 0.5, label='Average \n Test Accuracy', color='skyblue')
    graph2 = ax.bar(X_axis + 0.2, losses, 0.4, label='Average \n Test Loss', color='khaki')
    graph3 = ax2.bar(X_axis - 0.2, params, 0.4, label='Total Communicated \n Parameters', color='lightgreen')

    i = 0
    for p in graph3:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax2.text(x + width / 2,
                 y + height * 1.01,
                 str(params[i]),
                 ha='center',
                 weight='bold',
                 fontsize=10)
        i += 1

    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2,
                 y + height * 1.01,
                 str(round(accs[i] * 100, 2)) + '%',
                 ha='center',
                 weight='bold',
                 fontsize=10)
        i += 1

    i = 0
    for p in graph2:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2,
                 y + height * 1.01,
                 str(round(losses[i], 2)),
                 ha='center',
                 weight='bold',
                 fontsize=10)
        i += 1

    multicolor_ylabel(ax, ('Test Accuracy (%)', 'and', 'Test Loss'), ('y', 'k', 'b'), axis='y', fontsize=15)
    ax2.set_ylabel('Communicated Parameters', fontsize=15, color='g')
    ax2.set_yscale('symlog')

    legend = ax.legend(loc='best', ncol=3, fancybox=True, framealpha=0)

    ax.set_xticks(X_axis, X, rotation=20, ha='right')
    ax.set_ylim(0, 110)
    ax2.set_ylim(1, 1e8)
    ax2.set_yticks([0, 100, 10000, 1000000])
    # ax.tight_layout()
    # for t in legend.get_texts():
    #     t.set_ha('center')
    saveFilename = "fedavg-comparison.pdf"
    if save_fig:
        plt.savefig(saveFilename, format="pdf")
    else:
        plt.show()
    '''
