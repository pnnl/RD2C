import tensorflow as tf
import pandas as pd
import numpy as np


def darknet_data(rank, size, train_pct, train_bs, test_bs, coordination_size):
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
    coord_y = tf.convert_to_tensor(labels.iloc[:coordination_size])
    # coordination_set = tf.data.Dataset.from_tensor_slices((coord_x, coord_y)).batch(coordination_size)

    # get data info
    num_inputs = len(normalized_df.columns.to_list())
    num_outputs = len(traffic_categories)

    # Split training data amongst workers
    worker_data = np.array_split(normalized_df.iloc[coordination_size:, :], size)[rank]
    worker_label = np.array_split(labels.iloc[coordination_size:], size)[rank]

    # create train/test split
    num_data = len(worker_label)
    num_train = int(num_data * train_pct)
    # train
    train_x = tf.convert_to_tensor(worker_data.iloc[:num_train, :])
    train_y = tf.convert_to_tensor(worker_label.iloc[:num_train])
    train_set = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(train_bs)
    # test
    test_x = tf.convert_to_tensor(worker_data.iloc[num_train:, :])
    test_y = tf.convert_to_tensor(worker_label.iloc[num_train:])
    test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(test_bs)

    # non-iid Dataset
    # create train/test split
    nid_num_data = len(labels.iloc[coordination_size:])
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

    return train_set, test_set, coord_x, coord_y, nid_train_set, nid_test_x, nid_test_y, num_inputs, num_outputs