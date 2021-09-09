from __future__ import print_function, division
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import json
import os
import time
import traceback
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import Manager, Process
from client.bcosclient import BcosClient
from client.datatype_parser import DatatypeParser
from client.common.compiler import Compiler
from client.bcoserror import BcosException, BcosError
from client_config import client_config
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

tf.compat.v1.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# 序列化和反序列化
def serialize(data):
    json_data = json.dumps(data)
    return json_data


def deserialize(json_data):
    data = json.loads(json_data)
    return data


# 切分数据集
def split_data(clients_num):
    (X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # tf.random.set_seed(2345)
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)

    # 训练集划分给多个client
    X_train = np.array_split(X_train, clients_num)
    y_train = np.array_split(y_train, clients_num)

    return X_train, y_train, X_test, y_test


# 划分客户端数据集
CLIENT_NUM = 20
X_train, X_test, y_train, y_test = split_data(CLIENT_NUM)

manager = Manager()

# 节点角色常量
ROLE_TRAINER = "trainer"  # 训练节点
ROLE_COMM = "comm"  # 委员会

# 轮询的时间间隔，单位秒
QUERY_INTERVAL = 10

# 最大的执行轮次
MAX_EPOCH = 50 * CLIENT_NUM

# 设置模型
# n_features = 5
# n_class = 2

# 从文件加载abi定义
if os.path.isfile(client_config.solc_path) or os.path.isfile(client_config.solcjs_path):
    Compiler.compile_file("contracts/CommitteePrecompiled.sol")
abi_file = "contracts/CommitteePrecompiled.abi"
data_parser = DatatypeParser()
data_parser.load_abi_file(abi_file)
contract_abi = data_parser.contract_abi

# 定义合约地址
to_address = "0x0000000000000000000000000000000000005006"


# 写一个节点的工作流程
def run_one_node(node_id):
    """指定一个node id，并启动一个进程"""

    batch_size = 100
    learning_rate = 0.001
    trained_epoch = -1
    node_index = int(node_id.split('_')[-1])

    # 初始化bcos客户端
    try:
        client = BcosClient()
        # 为了更好模拟真实多个客户端场景，给不同的客户端分配不同的地址
        client.set_from_account_signer(node_id)
        print("{} initializing....".format(node_id))
    except Exception as e:
        client.finish()
        traceback.print_exc()

    def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        return x, y

    def local_training():

        print("{} begin training..".format(node_id))
        try:
            model, epoch = client.call(to_address, contract_abi, "QueryGlobalModel")
            model = deserialize(model)
            model_v = []
            for key in model:
                w_array = model[key]
                w_array = tf.Variable(w_array, tf.float32, name='{}'.format(key[0:-2]))
                model_v.append(w_array)

            conv_layers = [
                # unit 1, 64为channel数
                layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                # [8,8,64]
                layers.MaxPool2D(pool_size=[2, 2], strides=4, padding="same"),

                # unit 2, 128为channel数
                layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                # [2,2,128]
                layers.MaxPool2D(pool_size=[2, 2], strides=4, padding="same"),

                # unit 3, 256为channel数
                layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                # [1,1,256]
                layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
            ]
            fc_layers = [
                layers.Dense(100, activation=tf.nn.relu)
            ]

            # 选择第client_id客户端的数据集
            x = X_train[node_index]
            y = y_train[node_index]

            db = tf.data.Dataset.from_tensor_slices((x, y))
            db = db.map(preprocess).shuffle(1000).batch(batch_size)
            db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            db_test = db_test.map(preprocess).batch(batch_size)

            # 构建该客户端的cnn神经网络
            conv_net = Sequential(conv_layers)
            fc_net = Sequential(fc_layers)
            conv_net.build(input_shape=[None, 32, 32, 3])
            fc_net.build(input_shape=[None, 256])
            optimizer = optimizers.Adam(1e-5)

            variables = model_v

            # 转为tf数据格式
            for i in range(len(variables)):
                variables[i] = tf.cast(variables[i], dtype=tf.float32)

            # 将用全局更新好的权重更新该客户端模型的权重
            item = 0
            for i in range(len(variables)):
                if i < len(conv_net.trainable_variables):
                    tf.compat.v1.assign(conv_net.trainable_variables[i], variables[i])
                else:
                    tf.compat.v1.assign(fc_net.trainable_variables[item], variables[i])
                    item += 1
            # 连接卷积层和全连接层的权重，用于tf追踪变量，进行反向传播更新权重
            variables = conv_net.trainable_variables + fc_net.trainable_variables

            # 客户端的样本数
            n_samples = X_train[node_index].shape[0]

            # 开始训练
            for step, (x, y) in enumerate(db):
                with tf.GradientTape() as tape:
                    # [b, 32, 32, 3] => [b, 1, 1, 512]
                    out = conv_net(x)
                    # flatten => [b, 512]
                    out = tf.reshape(out, [-1, 256])
                    # [b, 512] => [b, 100]
                    logits = fc_net(out)
                    # one-hot [b] => [b, 100]
                    y_one_hot = tf.one_hot(y, depth=100)
                    # compute loss
                    loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(grads, variables))

            # test
            total_correct = 0
            total_num = 0
            for x, y in db_test:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = conv_net(x)
                # flatten => [b, 512]
                out = tf.reshape(out, [-1, 256])
                # [b, 512] => [b, 100]
                logits = fc_net(out)
                # logits => prob [b, 100]
                prob = tf.nn.softmax(logits, axis=1)
                # [b, 10] => [b]
                pred = tf.argmax(prob, axis=1)
                pred = tf.cast(pred, dtype=tf.int32)
                correct = tf.equal(pred, y)
                correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

                total_correct += correct
                total_num += x.shape[0]
            acc = total_correct / total_num
            print("acc:", float(acc))

            delta_model = conv_net.trainable_variables + fc_net.trainable_variables

            # 上传权重到区块链中
            variables_dict = {}
            for i in range(len(variables)):
                variables_dict[variables[i].name] = variables[i].numpy().tolist()
            delta_model = serialize(variables_dict)
            # meta = {'n_samples': n_samples, 'avg_cost': avg_cost}
            meta = {'n_samples': 1, 'avg_cost': 1}
            update_model = {'delta_model': delta_model, 'meta': meta}
            receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "UploadLocalUpdate", [update_model, epoch])

            nonlocal trained_epoch
            trained_epoch = epoch

        except Exception as e:
            client.finish()
            traceback.print_exc()

        return

    def local_testing(model_v):
        conv_layers = [
            # unit 1, 64为channel数
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            # [8,8,64]
            layers.MaxPool2D(pool_size=[2, 2], strides=4, padding="same"),

            # unit 2, 128为channel数
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            # [2,2,128]
            layers.MaxPool2D(pool_size=[2, 2], strides=4, padding="same"),

            # unit 3, 256为channel数
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            # [1,1,256]
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
        ]
        fc_layers = [
            layers.Dense(100, activation=tf.nn.relu)
        ]

        db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        db_test = db_test.map(preprocess).batch(batch_size)

        # 构建该客户端的cnn神经网络
        conv_net = Sequential(conv_layers)
        fc_net = Sequential(fc_layers)
        conv_net.build(input_shape=[None, 32, 32, 3])
        fc_net.build(input_shape=[None, 256])
        optimizer = optimizers.Adam(1e-5)

        variables = model_v

        # 转为tf数据格式
        for i in range(len(variables)):
            variables[i] = tf.cast(variables[i], dtype=tf.float32)

        # 将用全局更新好的权重更新该客户端模型的权重
        item = 0
        for i in range(len(variables)):
            if i < len(conv_net.trainable_variables):
                tf.compat.v1.assign(conv_net.trainable_variables[i], variables[i])
            else:
                tf.compat.v1.assign(fc_net.trainable_variables[item], variables[i])
                item += 1
        # 连接卷积层和全连接层的权重，用于tf追踪变量，进行反向传播更新权重
        variables = conv_net.trainable_variables + fc_net.trainable_variables

        # test
        total_correct = 0
        total_num = 0
        for x, y in db_test:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            out = conv_net(x)
            # flatten => [b, 512]
            out = tf.reshape(out, [-1, 256])
            # [b, 512] => [b, 100]
            logits = fc_net(out)
            # logits => prob [b, 100]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b]
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += correct
            total_num += x.shape[0]
        acc = total_correct / total_num

        return acc

    def local_scoring():
        try:
            res = client.call(to_address, contract_abi, "QueryAllUpdates")
            updates = res[0]

            # 如果获取不了任何更新信息，则认为还不能开始评分
            if len(updates) == 0:
                return

            updates = deserialize(updates)

            model, epoch = client.call(to_address, contract_abi, "QueryGlobalModel")
            model = deserialize(model)

            model_v = []
            for key in model:
                w_array = model[key]
                w_array = tf.Variable(w_array, tf.float32, name='{}'.format(key[0:-2]))
                model_v.append(w_array)

            print("{} begin scoring..".format(node_id))
            scores = {}
            for trainer_id, update in updates.items():
                update = deserialize(update)
                delta_model, meta = update['delta_model'], update['meta']
                update_v = []
                for key in model:
                    w_array = delta_model[key]
                    w_array = tf.Variable(w_array, tf.float32, name='{}'.format(key[0:-2]))
                    update_v.append(w_array)
                scores[trainer_id] = local_testing(update_v)

            receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "UploadScores",
                                                          [epoch, serialize(scores)])

            nonlocal trained_epoch
            trained_epoch = epoch

        except Exception as e:
            client.finish()
            traceback.print_exc()

        return

    def wait():
        time.sleep(random.uniform(QUERY_INTERVAL, QUERY_INTERVAL * 3))
        return

    def main_loop():

        # 注册节点
        try:
            receipt = client.sendRawTransactionGetReceipt(to_address, contract_abi, "RegisterNode", [])
            print("{} registered successfully".format(node_id))

            while True:

                # 查询对应节点的角色和当前迭代次数
                (role, global_epoch) = client.call(to_address, contract_abi, "QueryState")

                # print("{} role: {}, local e: {}, up_c: {}, sc_c: {}"\
                #     .format(node_id, role, trained_epoch,
                #             update_count, score_count))

                if global_epoch > MAX_EPOCH:
                    break

                if global_epoch <= trained_epoch:
                    # print("{} skip.".format(node_id))
                    wait()
                    continue

                if role == ROLE_TRAINER:
                    local_training()

                if role == ROLE_COMM:
                    local_scoring()

                wait()

        except Exception as e:
            client.finish()
            traceback.print_exc()

        return

    main_loop()

    # 关闭客户端
    client.finish()


# 发起人的全局测试
def run_sponsor():
    test_epoch = 0

    def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        return x, y

    # 跑测试集
    def global_testing(model_v):
        conv_layers = [
            # unit 1, 64为channel数
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            # [8,8,64]
            layers.MaxPool2D(pool_size=[2, 2], strides=4, padding="same"),

            # unit 2, 128为channel数
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            # [2,2,128]
            layers.MaxPool2D(pool_size=[2, 2], strides=4, padding="same"),

            # unit 3, 256为channel数
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            # [1,1,256]
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
        ]
        fc_layers = [
            layers.Dense(100, activation=tf.nn.relu)
        ]

        batch_size = 100

        db_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        db_test = db_test.map(preprocess).batch(batch_size)

        # 构建该客户端的cnn神经网络
        conv_net = Sequential(conv_layers)
        fc_net = Sequential(fc_layers)
        conv_net.build(input_shape=[None, 32, 32, 3])
        fc_net.build(input_shape=[None, 256])
        optimizer = optimizers.Adam(1e-5)

        variables = model_v

        # 转为tf数据格式
        for i in range(len(variables)):
            variables[i] = tf.cast(variables[i], dtype=tf.float32)

        # 将用全局更新好的权重更新该客户端模型的权重
        item = 0
        for i in range(len(variables)):
            if i < len(conv_net.trainable_variables):
                tf.compat.v1.assign(conv_net.trainable_variables[i], variables[i])
            else:
                tf.compat.v1.assign(fc_net.trainable_variables[item], variables[i])
                item += 1
        # 连接卷积层和全连接层的权重，用于tf追踪变量，进行反向传播更新权重
        variables = conv_net.trainable_variables + fc_net.trainable_variables

        # test
        total_correct = 0
        total_num = 0
        for x, y in db_test:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            out = conv_net(x)
            # flatten => [b, 512]
            out = tf.reshape(out, [-1, 256])
            # [b, 512] => [b, 100]
            logits = fc_net(out)
            # logits => prob [b, 100]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b]
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += correct
            total_num += x.shape[0]
        acc = total_correct / total_num

        return acc

    def wait():
        time.sleep(random.uniform(QUERY_INTERVAL, QUERY_INTERVAL * 3))
        return

    def test():
        # 初始化bcos客户端
        try:
            client = BcosClient()

            while True:
                model, epoch = client.call(to_address, contract_abi, "QueryGlobalModel")
                model = deserialize(model)

                model_v = []
                for key in model:
                    w_array = model[key]
                    w_array = tf.Variable(w_array, tf.float32, name='{}'.format(key[0:-2]))
                    model_v.append(w_array)

                nonlocal test_epoch
                if epoch > test_epoch:
                    test_acc = global_testing(model_v)
                    print("Epoch: {:03}, test_acc: {:.4f}" \
                          .format(test_epoch, test_acc))
                    test_epoch = epoch

                wait()

        except Exception as e:
            client.finish()
            traceback.print_exc()

    test()

    # 关闭客户端
    client.finish()


if __name__ == "__main__":
    process_pool = []
    for i in range(CLIENT_NUM):
        node_id = 'node_{}'.format(len(process_pool))
        p = Process(target=run_one_node, args=(node_id,))
        p.start()
        process_pool.append(p)
        time.sleep(3)

    # 加入发起者的进行全局测试
    p = Process(target=run_sponsor)
    p.start()
    process_pool.append(p)

    for p in process_pool:
        p.join()
