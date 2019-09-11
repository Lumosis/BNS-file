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

import numpy as np
from ..util.log import *
import pickle as pk

def get_rollout(env, policy, render):
    # obs, done = np.array(env.reset()), False
    obs = np.array(env.reset())
    while obs.any() == None:
        obs = np.array(env.reset())
        print('None!!!')
    done = False
    rollout = []
    # i = 0
    while not done:
        # Render
        

        # Action
        act = policy.predict(np.array([obs]))
        # print(act)
        act = act[0]
        # print(act)
        # input(type(act))

        # Step
        next_obs, rew, done = env.step(act)

        # Rollout (s, a, r)
        rollout.append((obs, act, sum(rew[0])))

        # Update (and remove LazyFrames)
        obs = np.array(next_obs)
        # i += 1
        # if i ==10:
        #     break


    return rollout

def get_rollouts(env, policy, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy, render))
    # input(rollouts)
    return rollouts

def _sample(obss, acts, qs, max_pts, is_reweight):
    # Step 1: Compute probabilities
    ps = []
    for i in qs:
        temp = np.mean(i[(-i).argsort()[:5]]) - np.mean(i[(i).argsort()[:5]])
        ps.append(temp)
    ps = np.array(ps)
    ps = ps / np.sum(ps)


    # Step 2: Sample points
    # print(len(obss), len(acts), len(qs), len(ps), np.sum(ps>0))
    # input('sample')
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False)    

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx]

class TransformerPolicy:
    def __init__(self, policy, state_transformer):
        self.policy = policy
        self.state_transformer = state_transformer

    def predict(self, obss):
        # input("get!!")
        ret = self.policy.predict(np.array([self.state_transformer(obs) for obs in obss]))
        # print(ret.shape)
        # print(type(ret))
        
        ret = ret.tolist()
        # input('here')
        # input(ret)
        return ret

def test_policy(env, policy, state_transformer, n_test_rollouts):
    wrapped_student = TransformerPolicy(policy, state_transformer)
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout(env, wrapped_student, False)
        cum_rew += sum((rew for _, _, rew in student_trace))
    return cum_rew / n_test_rollouts

def identify_best_policy(env, policies, state_transformer, n_test_rollouts):
    log('Initial policy count: {}'.format(len(policies)), INFO)
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1)/2)
        log('Current policy count: {}'.format(n_policies), INFO)

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env, policy, state_transformer, n_test_rollouts)
            new_policies.append((policy, new_rew))
            log('Reward update: {} -> {}'.format(rew, new_rew), INFO)

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0]

def _get_action_sequences_helper(trace, seq_len):
    acts = [act for _, act, _ in trace]
    seqs = []
    for i in range(len(acts) - seq_len + 1):
        seqs.append(acts[i:i+seq_len])
    return seqs

def get_action_sequences(env, policy, seq_len, n_rollouts):
    # Step 1: Get action sequences
    seqs = []
    for _ in range(n_rollouts):
        trace = get_rollout(env, policy, False)
        seqs.extend(_get_action_sequences_helper(trace, seq_len))

    # Step 2: Bin action sequences
    counter = {}
    for seq in seqs:
        s = str(seq)
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    # Step 3: Sort action sequences
    seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])

    return seqs_sorted

def train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts):
    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []
    wrapped_student = TransformerPolicy(student, state_transformer)
    n_batch_rollouts = 1
    # Step 1: Generate some supervised traces into the buffer
    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    obss.extend((state_transformer(obs) for obs, _, _ in trace))

    acts.extend((act for _, act, _ in trace))
    qs.extend(teacher.predict_q(np.array([obs for obs, _, _ in trace])))

    # input(obss)
    # input(acts)
    # input(qs)
    # Step 2: Dagger outer loop
    training = []
    testing = []
    rew = []
    tm_num = []
    for i in range(max_iters):
        log('Iteration {}/{}'.format(i, max_iters), INFO)

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(np.array(obss), np.array(acts), np.array(qs), max_samples, is_reweight)
        log('Training student with {} points'.format(len(cur_obss)), INFO)
        tm_num.append(len(cur_obss))
        # print('cur_obss')
        # print(cur_obss[0])
        # print(type(cur_obss))
        # print(cur_obss.shape)
        # input()
        training_accuracy, test_accuracy = student.train(cur_obss, cur_acts, train_frac)
        training.append(training_accuracy)
        testing.append(test_accuracy)


        # Step 2b: Generate trace using student
        student_trace = get_rollouts(env, wrapped_student, False, n_batch_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]
        
        # Step 2c: Query the oracle for supervision
        teacher_qs = teacher.predict_q(student_obss) # at the interface level, order matters, since teacher.predict may run updates
        teacher_acts = teacher.predict(student_obss)
        # print(student_obss)
        # print(teacher_qs)
        # print(teacher_acts)
        # input('train!')

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((state_transformer(obs) for obs in student_obss))
        acts.extend(teacher_acts)
        # input('train_dagger')
        # input(len(obss))
        # input(len(acts))
        qs.extend(teacher_qs)

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew), INFO)
        rew.append(cur_rew)
        students.append((student.clone(), cur_rew))

    f = open('./save/train_data.pk', 'wb')
    pk.dump(training, f)
    f.close()
    f = open('./save/test_data.pk', 'wb')
    pk.dump(testing, f)
    f.close()
    f = open('./save/reward.pk', 'wb')
    pk.dump(rew, f)
    f.close()
    f = open('./save/tm_num.pk', 'wb')
    pk.dump(tm_num, f)
    f.close()
    max_student = identify_best_policy(env, students, state_transformer, n_test_rollouts)

    return max_student
