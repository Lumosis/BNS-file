# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle as pk
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from ..util.log import *

def cal_accu(a, b):
    
    accu_1 = 0
    accu_2 = 0
    for i in range(len(a)):
        flag = True
        temp = list(set(a[i]))
        for j in temp:
            if j in b[i]:
                accu_1 += 1
            else:
                flag = False
        if flag:
            accu_2 += 1

    accu_1 = accu_1 / (len(a) * 5)
    accu_2 = accu_2 / (len(a)) 
    return accu_1, accu_2      


def accuracy(policy, obss, acts):
    # print(np.sort(acts))
    # print(np.sort(policy.predict(obss)))
    # print(np.sort(acts) == np.sort(policy.predict(obss)))
    
    # print(np.sort(acts))
    # print(np.sort(policy.predict(obss)))
    # input('accu!!!!!!!!!!!!')
    accu_1, accu_2 = cal_accu(acts.tolist(), policy.predict(obss).tolist())
    return accu_1, accu_2

def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test

def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()

def save_dt_policy_viz(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    export_graphviz(dt_policy.tree, dirname + '/' + fname)

def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy

class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        # print('train_obss')
        # print(obss_train[0])
        # print(type(obss_train))
        # print(obss_train.shape)
        # input()
        self.fit(obss_train, acts_train)
        training_accuracy = accuracy(self, obss_train, acts_train)
        test_accuracy = accuracy(self, obss_test, acts_test)
        log('Train accuracy: {}'.format(training_accuracy), INFO)
        log('Test accuracy: {}'.format(test_accuracy), INFO)
        log('Number of nodes: {}'.format(self.tree.tree_.node_count), INFO)
        # input('hello!')
        return training_accuracy, test_accuracy

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone

