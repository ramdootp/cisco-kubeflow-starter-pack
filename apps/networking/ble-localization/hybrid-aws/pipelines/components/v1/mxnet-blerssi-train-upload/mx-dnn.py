from __future__ import print_function
import os
import json
import boto3
import tarfile
import pickle
import argparse

# non-standard libraries
from clize import Parameter, run
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import pandas as pd


def flatten(tarinfo):
    tarinfo.name = os.path.basename(tarinfo.name)
    return tarinfo


def train(*train: Parameter.REQUIRED,
          bucket_name='mxnet-model-store',
          data_path=''):
    """Trains and uploads the BLERSSI model to S3
    
    :param bucket_name (string): the bucket to upload the model to.
    :param data_path (string): the path to the dataset.
    """
    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    data_ctx = ctx
    model_ctx = ctx

    batch_size = 64
    num_inputs = 13
    num_outputs = 105
    num_examples = 1136

    df_full = pd.read_csv(data_path +
                          'iBeacon_RSSI_Labeled.csv')  #Labeled dataset
    COLUMNS = list(df_full.columns)
    FEATURES = COLUMNS[2:]

    # Input Data Preprocessing
    df_full = df_full.drop(['date'], axis=1)

    #Output Data Preprocessing
    dict = {
        'O02': 0,
        'P01': 1,
        'P02': 2,
        'R01': 3,
        'R02': 4,
        'S01': 5,
        'S02': 6,
        'T01': 7,
        'U02': 8,
        'U01': 9,
        'J03': 10,
        'K03': 11,
        'L03': 12,
        'M03': 13,
        'N03': 14,
        'O03': 15,
        'P03': 16,
        'Q03': 17,
        'R03': 18,
        'S03': 19,
        'T03': 20,
        'U03': 21,
        'U04': 22,
        'T04': 23,
        'S04': 24,
        'R04': 25,
        'Q04': 26,
        'P04': 27,
        'O04': 28,
        'N04': 29,
        'M04': 30,
        'L04': 31,
        'K04': 32,
        'J04': 33,
        'I04': 34,
        'I05': 35,
        'J05': 36,
        'K05': 37,
        'L05': 38,
        'M05': 39,
        'N05': 40,
        'O05': 41,
        'P05': 42,
        'Q05': 43,
        'R05': 44,
        'S05': 45,
        'T05': 46,
        'U05': 47,
        'S06': 48,
        'R06': 49,
        'Q06': 50,
        'P06': 51,
        'O06': 52,
        'N06': 53,
        'M06': 54,
        'L06': 55,
        'K06': 56,
        'J06': 57,
        'I06': 58,
        'F08': 59,
        'J02': 60,
        'J07': 61,
        'I07': 62,
        'I10': 63,
        'J10': 64,
        'D15': 65,
        'E15': 66,
        'G15': 67,
        'J15': 68,
        'L15': 69,
        'R15': 70,
        'T15': 71,
        'W15': 72,
        'I08': 73,
        'I03': 74,
        'J08': 75,
        'I01': 76,
        'I02': 77,
        'J01': 78,
        'K01': 79,
        'K02': 80,
        'L01': 81,
        'L02': 82,
        'M01': 83,
        'M02': 84,
        'N01': 85,
        'N02': 86,
        'O01': 87,
        'I09': 88,
        'D14': 89,
        'D13': 90,
        'K07': 91,
        'K08': 92,
        'N15': 93,
        'P15': 94,
        'I15': 95,
        'S15': 96,
        'U15': 97,
        'V15': 98,
        'S07': 99,
        'S08': 100,
        'L09': 101,
        'L08': 102,
        'Q02': 103,
        'Q01': 104
    }

    df_full['location'] = df_full['location'].map(dict)
    df_train = df_full.sample(frac=0.8, random_state=200)
    df_valid = df_full.drop(df_train.index)

    df_X = df_train[FEATURES]
    df_y = df_train['location']
    t_X = df_valid[FEATURES]
    t_y = df_valid['location']

    X_np = df_X.to_numpy()
    y_np = df_y.to_numpy()
    X = nd.array(X_np)
    y = nd.array(y_np)
    test_X = nd.array(t_X.to_numpy())
    test_y = nd.array(t_y.to_numpy())

    print(X, y)

    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                       batch_size=batch_size,
                                       shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(test_X, test_y),
                                      batch_size=batch_size,
                                      shuffle=True)
    print(train_data)
    num_hidden = [13, 65, 110]
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(13, activation="relu"))
        net.add(gluon.nn.Dense(65, activation="relu"))
        net.add(gluon.nn.Dense(110, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))

    net.hybridize()

    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': .01})

    epochs = 100

    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("########################################")
        print(
            "Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
            (e, cumulative_loss / num_examples, train_accuracy, test_accuracy))

    if not os.path.exists('model'):
        os.mkdir('model')

    net.export("model/blenet", epoch=100)
    print(net.collect_params())
    trainer.save_states('model/blenet-0100.states')

    tar = tarfile.open("model.tar.gz", "w:gz")
    tar.add("model", filter=flatten)
    tar.close()

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    response = s3_client.upload_file("model.tar.gz", bucket_name,
                                     "blerssi/model.tar.gz")


if __name__ == '__main__':
    run(train)
