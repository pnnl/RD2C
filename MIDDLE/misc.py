import numpy as np
import os


class Recorder(object):
    def __init__(self, name, size, rank, graph_type, epochs, l3, coord_size, outputPath, save_folder_name=None):
        self.epoch_time = list()
        self.comp_time = list()
        self.comm_time = list()
        self.train_loss = list()
        self.train_acc = list()
        self.test_loss = list()
        self.test_acc = list()
        self.rank = rank
        self.size = size
        if save_folder_name is None:
            self.saveFolderName = outputPath + '/' + name + '-' + str(size) + 'Worker-' + str(epochs) + 'Epochs-' + \
                              str(l3) + 'L3Penalty-' + str(coord_size) + 'Csize-' + str(graph_type)
        else:
            self.saveFolderName = save_folder_name

        if rank == 0 and not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)

    def add_to_file(self, epoch_time, comp_time, comm_time, train_loss, train_acc, test_loss, test_acc):
        self.epoch_time.append(epoch_time)
        self.comp_time.append(comp_time)
        self.comm_time.append(comm_time)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.epoch_time, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comp-time.log', self.comp_time, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comm-time.log', self.comm_time, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-loss.log', self.train_loss, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc.log', self.train_acc, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-loss.log', self.test_loss, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc.log', self.test_acc, delimiter=',')
