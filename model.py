'''
rnn attention on env, turnover and profile, separate bias
'''
from collections import OrderedDict
from itertools import product
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

from myutil import time_estimator
from nn_util import get_train_test_split_indicies, get_batch_dict


def prepare_data(create_new=False, ratio=None, base_path='../data/temp_output/nn_data/'):
    import pickle
    from os.path import exists, join

    if not create_new and exists(join(base_path, 'ids')) \
            and exists(join(base_path, 'turnover_seq')) \
            and exists(join(base_path, 'labels')) \
            and exists(join(base_path, 'env_seq')) \
            and exists(join(base_path, 'profile')):
        print('load existing')
        ids = pickle.load(open(join(base_path, 'ids'), 'rb'))
        sequence_features = pickle.load(
            open(join(base_path, 'turnover_seq'), 'rb'))
        env_sequence_features = pickle.load(
            open(join(base_path, 'env_seq'), 'rb'))
        profile_features = pickle.load(open(join(base_path, 'profile'), 'rb'))
        labels = pickle.load(open(join(base_path, 'labels'), 'rb'))
    else:
        print('not found, constructing new')
        seq_d = pd.read_pickle(join(base_path, 'filtered_seq_with_profile_brief'))

        seq_d.reset_index(inplace=True)

        ids = seq_d.apply(lambda row: '{}_{}_{}'.format(
            row.uid, row.leave_date_str, row.observe_start), axis=1).tolist()

        labels = np.zeros((len(seq_d), 2))
        labels_indicies = seq_d.leave_date_str.str.contains(
            'NaT').astype(int) - 1
        labels[range(len(seq_d)), labels_indicies] = 1

        sequence_features = seq_d.turnover_seq.tolist()

        env_sequence_features = seq_d.env_seq

        profile_features = np.array(seq_d.profile.tolist())

        del seq_d

        pickle.dump(ids, open(join(base_path, 'ids'), 'wb'))
        pickle.dump(sequence_features, open(
            join(base_path, 'turnover_seq'), 'wb'))
        pickle.dump(env_sequence_features, open(
            join(base_path, 'env_seq'), 'wb'))
        pickle.dump(profile_features, open(join(base_path, 'profile'), 'wb'))
        pickle.dump(labels, open(join(base_path, 'labels'), 'wb'))

    if ratio:
        label_vec = np.argmax(labels, axis=1)

        n_positive = np.sum(label_vec == 1)
        n_negative = len(label_vec) - n_positive

        if n_positive / n_negative > ratio:
            # sample positive
            sample_label = 1
            sample_num = int(n_negative * ratio)
        else:
            # sample negative
            sample_label = 0
            sample_num = int(n_positive / ratio)

        l1 = np.random.choice(np.where(label_vec == sample_label)[
                                  0], sample_num).tolist()
        l2 = np.where(label_vec == (1 - sample_label))[0].tolist()
        idxes = sorted(l1 + l2)

        ids = [ids[i] for i in idxes]
        sequence_features = [sequence_features[i] for i in idxes]
        env_sequence_features = [env_sequence_features[i] for i in idxes]
        profile_features = profile_features[idxes, :]

        labels = labels[idxes, :]
        labels_vec = np.argmax(labels, axis=1)
        n_positive_adjusted = np.sum(labels_vec == 1)
        n_negative_adjusted = np.sum(labels_vec == 0)
        print('# positive samples:', n_positive_adjusted)
        print('# negative samples:', n_negative_adjusted)
        print('# total:', len(labels_vec))
        print('ratio:', n_positive_adjusted / n_negative_adjusted)

    return {
        'ids': ids,
        'sequence_features': sequence_features,
        'env_sequence_features': env_sequence_features,
        'profile': profile_features,
        'labels': labels
    }


def attention(turnover_input, env_input, profile_input, attention_size, initializer, attention_mask, time_major=False,
              return_alphas=False):
    # Trainable parameters
    w_omegas_turnover = tf.get_variable('w_omegas_turnover', shape=[turnover_input.shape[
                                                                        2].value, attention_size], dtype=tf.float32,
                                        initializer=initializer())
    w_omegas_env = tf.get_variable('w_omegas_env', shape=[env_input.shape[
                                                              2].value, attention_size], dtype=tf.float32,
                                   initializer=initializer())
    w_omegas_profile = tf.get_variable('w_omegas_profile', shape=[profile_input.shape[
                                                                      1].value, attention_size], dtype=tf.float32,
                                       initializer=initializer())

    b_omega_turnover = tf.get_variable('b_omega_turnover', shape=[
        attention_size], dtype=tf.float32, initializer=initializer())
    b_omega_env = tf.get_variable('b_omega_env', shape=[
        attention_size], dtype=tf.float32, initializer=initializer())
    b_omega_profile = tf.get_variable('b_omega_profile', shape=[
        attention_size], dtype=tf.float32, initializer=initializer())

    u_omega = tf.get_variable(
        'u_omega', shape=[attention_size], dtype=tf.float32, initializer=initializer())

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size

        # v = tf.tanh(seq_concat + b_omega)
        projected_turnover = tf.tensordot(
            turnover_input, w_omegas_turnover, axes=1) + b_omega_turnover
        projected_env = tf.tensordot(env_input, w_omegas_env, axes=1) + b_omega_env
        projected_profile = tf.reshape(tf.tensordot(
            profile_input, w_omegas_profile, axes=1), (-1, 1, attention_size)) + b_omega_profile
        projected = tf.concat(
            [projected_env, projected_turnover, projected_profile], axis=1)

        v_turnover = tf.tanh(projected_turnover)
        v_env = tf.tanh(projected_env)
        v_profile = tf.tanh(projected_profile)
        v = tf.concat([v_env, v_turnover, v_profile], axis=1)

    # For each of the timestamps its vector of size A from `v` is reduced with
    # `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape

    masked_alphas = tf.multiply(attention_mask, tf.exp(tf.multiply(attention_mask, vu)))
    alphas = tf.divide(masked_alphas, tf.reshape(tf.reduce_sum(masked_alphas, axis=1), (-1, 1)), name='alphas')

    # Output of (Bi-)RNN is reduced with attention vector; the result has
    # (B,D) shape
    output = tf.reduce_sum(projected * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas, vu


def build_graph(params):
    tf.reset_default_graph()
    global_step = tf.Variable(1, name='global_step', trainable=False)
    N_CLASSES = params['N_CLASSES']

    attention_mask = tf.placeholder(tf.float32, shape=[None, None])

    target_label = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    LEARNING_RATE = params['LEARNING_RATE']

    # construct peer turnover rnn(dynamic) model
    N_SEQUENCE_FEATUREN = params['N_SEQUENCE_FEATUREN']
    N_ENV_SEQUENCE_FEATUREN = params['N_ENV_SEQUENCE_FEATUREN']
    SEQ_DENSE_ACT_FUNC = params['SEQ_DENSE_ACT_FUNC']
    NUM_OF_RNN_CELLS = params['NUM_OF_RNN_CELLS']
    DIM_OF_TURNOVER_RNN_OUTPUT = params['DIM_OF_TURNOVER_RNN_OUTPUT']
    DIM_OF_ENV_RNN_OUTPUT = params['DIM_OF_ENV_RNN_OUTPUT']
    N_PROFILE_FEATUREN = params['N_PROFILE_FEATUREN']
    INITIALIZER = params['INITIALIZER']

    sequence_input_layer = tf.placeholder(
        tf.float32, shape=[None, None, N_SEQUENCE_FEATUREN])

    # if params['BI_DIRECTION_RNN']:
    #     # bi-direction
    #     f_rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in [
    #         DIM_OF_RNN_OUTPUT] * NUM_OF_RNN_CELLS]
    #     f_multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(f_rnn_layers)
    #     b_rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in [
    #         DIM_OF_RNN_OUTPUT] * NUM_OF_RNN_CELLS]
    #     b_multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(b_rnn_layers)
    #     bi_outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_multi_rnn_cell,
    #                                                         cell_bw=b_multi_rnn_cell,
    #                                                         inputs=sequence_input_layer,
    #                                                         dtype=tf.float32,
    #                                                         time_major=False)
    #     outputs = tf.concat(bi_outputs, 2)
    # else:
    # one direction
    if NUM_OF_RNN_CELLS > 1:
        rnn_layers = [tf.contrib.rnn.LSTMCell(
            DIM_OF_TURNOVER_RNN_OUTPUT, initializer=INITIALIZER()) for i in range(NUM_OF_RNN_CELLS)]
        multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
    else:
        multi_rnn_cell = tf.contrib.rnn.LSTMCell(
            DIM_OF_TURNOVER_RNN_OUTPUT, initializer=INITIALIZER())

    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=sequence_input_layer,
                                       dtype=tf.float32,
                                       time_major=False,
                                       scope='turnover_rnn')

    # construct env turnover rnn
    env_seq_input_layer = tf.placeholder(
        tf.float32, shape=[None, None, N_ENV_SEQUENCE_FEATUREN])

    # if params['BI_DIRECTION_RNN']:
    #     # bi-direction
    #     f_env_rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in [
    #         DIM_OF_RNN_OUTPUT] * NUM_OF_RNN_CELLS]
    #     f_env_multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(f_env_rnn_layers)
    #     b_evn_rnn_layers = [tf.contrib.rnn.LSTMCell(size) for size in [
    #         DIM_OF_RNN_OUTPUT] * NUM_OF_RNN_CELLS]
    #     b_env_multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(b_evn_rnn_layers)
    #     env_bi_outputs, env_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_env_multi_rnn_cell,
    #                                                                 cell_bw=b_env_multi_rnn_cell,
    #                                                                 inputs=env_seq_input_layer,
    #                                                                 dtype=tf.float32,
    #                                                                 time_major=False)
    #     env_outputs = tf.concat(env_bi_outputs, 2)
    # else:
    # one direction
    if NUM_OF_RNN_CELLS > 1:
        env_rnn_layers = [tf.contrib.rnn.LSTMCell(
            DIM_OF_ENV_RNN_OUTPUT, initializer=INITIALIZER()) for i in range(NUM_OF_RNN_CELLS)]
        multi_env_rnn_cell = tf.contrib.rnn.MultiRNNCell(env_rnn_layers)
    else:
        multi_env_rnn_cell = tf.contrib.rnn.LSTMCell(
            DIM_OF_ENV_RNN_OUTPUT, initializer=INITIALIZER())

    env_outputs, env_state = tf.nn.dynamic_rnn(cell=multi_env_rnn_cell,
                                               inputs=env_seq_input_layer,
                                               dtype=tf.float32,
                                               time_major=False,
                                               scope='env_rnn')

    # construct profile
    profile_input_layer = tf.placeholder(
        tf.float32, shape=[None, N_PROFILE_FEATUREN])
    profile_dense_layer = tf.layers.dense(profile_input_layer, params[
        'PROFILE_DENSE_DIM'], activation=tf.nn.relu, kernel_initializer=INITIALIZER())

    with tf.name_scope('global_attention'):
        att_output, alphas, vu = attention(outputs, env_outputs, profile_dense_layer, params[
            'ATTENTION_SIZE'], INITIALIZER, attention_mask, return_alphas=True)

    seq_dropout_layer = tf.layers.dropout(
        att_output, params['SEQ_DROPOUT_PROB'])
    seq_dropout_layer.set_shape((None, params['ATTENTION_SIZE']))

    sequence_dense_layer = tf.layers.dense(seq_dropout_layer, params[
        'SEQ_DENSE_DIM'], activation=SEQ_DENSE_ACT_FUNC, kernel_initializer=INITIALIZER())

    logits = tf.layers.dense(sequence_dense_layer,
                             N_CLASSES, kernel_initializer=INITIALIZER())

    positive_probs = tf.nn.softmax(logits)[:, 1]
    positive_probs = tf.identity(positive_probs, name='softmax')
    pred_labels = tf.argmax(input=logits, axis=1, name='pred_labels')

    target_label_1d = tf.argmax(target_label, 1, name='target_label_1d')

    auc = tf.metrics.auc(target_label_1d, positive_probs, name='auc')

    correct_pred = tf.equal(tf.argmax(logits, 1),
                            tf.argmax(target_label, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32), name='accuracy')
    tf.summary.scalar('rnn_accuracy', accuracy)

    loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=target_label), name='loss')
    tf.summary.scalar('loss', loss_op)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(
        loss_op, name='train_op', global_step=global_step)
    merged = tf.summary.merge_all()

    return {
        'global_step': global_step,
        'target_label': target_label,
        'summary_merged': merged,
        'train_op': train_op,
        'loss_op': loss_op,
        'accuracy': accuracy,
        'pred_labels': pred_labels,
        'sequence_input_layer': sequence_input_layer,
        'env_sequence_input_layer': env_seq_input_layer,
        'profile_input_layer': profile_input_layer,
        'attention_mask_input': attention_mask,
        'auc': auc,
        'softmax': positive_probs,
        'logits': logits,
        'alphas': alphas,
        'vu': vu
    }


def train(graph, run_from_new, model_output_path, model_load_path, params, datas, train_idxes, validate_idxes):
    from myutil import auc
    TRAIN_STEPS = params['TRAIN_STEPS']
    RECORD_EVERY_N_ITERATIONS = params['RECORD_EVERY_N_ITERATIONS']
    VALIDATE_EVERY_N_ITERATIONS = params['VALIDATE_EVERY_N_ITERATIONS']
    BATCH_SIZE = params['BATCH_SIZE']

    # initializer = tf.global_variables_initializer()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    merged = graph['summary_merged']
    global_step = graph['global_step']
    target_label = graph['target_label']
    train_op = graph['train_op']
    loss_op = graph['loss_op']
    accuracy = graph['accuracy']
    sequence_input_layer = graph['sequence_input_layer']
    env_sequence_input_layer = graph['env_sequence_input_layer']
    profile_input_layer = graph['profile_input_layer']
    attention_mask_input = graph['attention_mask_input']

    saver = tf.train.Saver(max_to_keep=TRAIN_STEPS)

    with tf.Session() as sess:
        # restore model from file
        if run_from_new:
            accuracies = pd.DataFrame(
                columns=['step', 'train_accuracy', 'validate_accuracy', 'train_auc', 'validate_auc'])
            sess.run(init)
        else:
            accuracies = pd.read_csv(
                model_load_path + '_accuracies', encoding='utf8')
            saver.restore(
                sess, '{}-{}'.format(model_load_path, accuracies.step.max()))

        writer = tf.summary.FileWriter(params['LOG_DIR'], sess.graph)

        # prepare data for training/validate/test
        start_step = sess.run(global_step)

        n_validate = len(validate_idxes)

        # mask for attention
        profile_mask = np.full((n_validate, 1), 1)
        env_mask = np.full((n_validate, datas['env_sequence_features'][0].shape[0]), 1)
        lens = [len(datas['sequence_features'][idx]) for idx in validate_idxes]
        sequence_mask = np.zeros((n_validate, max(lens)))
        for i in range(n_validate):
            sequence_mask[i, -lens[i]:] = 1
        attention_mask = np.hstack([env_mask, sequence_mask, profile_mask])

        validate_feed_dict = {
            sequence_input_layer: tf.keras.preprocessing.sequence.pad_sequences(
                [datas['sequence_features'][idx] for idx in validate_idxes], dtype='float'),
            env_sequence_input_layer: tf.keras.preprocessing.sequence.pad_sequences(
                [datas['env_sequence_features'][idx] for idx in validate_idxes], dtype='float'),
            profile_input_layer: datas['profile'][validate_idxes],
            target_label: datas['labels'][validate_idxes],
            attention_mask_input: attention_mask
        }
        best_accuracy = 0
        for step in range(start_step, start_step + TRAIN_STEPS):

            datas_batch = get_batch_dict(datas, train_idxes, step, BATCH_SIZE)

            profile_mask = np.full((BATCH_SIZE, 1), 1)
            env_mask = np.full((BATCH_SIZE, datas['env_sequence_features'][0].shape[0]), 1)
            lens = [len(s) for s in datas_batch['sequence_features']]
            sequence_mask = np.zeros((BATCH_SIZE, max(lens)))
            for i in range(BATCH_SIZE):
                sequence_mask[i, -lens[i]:] = 1
            attention_mask = np.hstack([env_mask, sequence_mask, profile_mask])

            train_feed_dict = {
                sequence_input_layer: tf.keras.preprocessing.sequence.pad_sequences(datas_batch['sequence_features'],
                                                                                    dtype='float'),
                env_sequence_input_layer: tf.keras.preprocessing.sequence.pad_sequences(
                    datas_batch['env_sequence_features'], dtype='float'),
                profile_input_layer: datas_batch['profile'],
                target_label: datas_batch['labels'],
                attention_mask_input: attention_mask
            }

            summary, _, loss, train_accuracy, train_softmax = sess.run(
                [merged, train_op, loss_op, accuracy, graph['softmax']], train_feed_dict)
            train_auc = auc(
                np.argmax(datas_batch['labels'], axis=1), train_softmax)

            writer.add_summary(summary, step)

            if step % VALIDATE_EVERY_N_ITERATIONS == 0:

                validate_accuracy, validate_softmax = sess.run(
                    [accuracy, graph['softmax']], validate_feed_dict)
                validate_auc = auc(
                    np.argmax(datas['labels'][validate_idxes], axis=1), validate_softmax)

                if step % 50 == 0:
                    print('step {} train_acc {:.3f} validate_acc {:.3f} train_auc:{:.3f} validate_auc:{:.3f}'.format(
                        step, train_accuracy, validate_accuracy, validate_accuracy, validate_auc))

                accuracies.loc[len(accuracies)] = [
                    step, train_accuracy, validate_accuracy, train_auc, validate_auc]

                if validate_accuracy > best_accuracy or step % RECORD_EVERY_N_ITERATIONS == 0:
                    saver.save(sess, model_output_path, global_step=step)

                if validate_accuracy > best_accuracy:
                    best_accuracy = validate_accuracy

                accuracies.to_csv(model_load_path + '_accuracies',
                                  index=False, encoding='utf8')

        writer.close()
        accuracies.to_csv(model_load_path + '_accuracies',
                          index=False, encoding='utf8')
        print('best training performance on validate data:')
        print(accuracies.loc[accuracies.validate_accuracy.idxmax()])


def test(params, datas, test_idxes, model_load_path, step=None, use_validate=True):
    from myutil import auc

    graph = build_graph(params)

    if step:
        best_model_step = step
    else:
        accuracies = pd.read_csv(
            model_load_path + '_accuracies', encoding='utf8')
        best_model_step = accuracies.step.loc[
            accuracies.validate_accuracy.idxmax()]

    sequence_input_layer = graph['sequence_input_layer']
    env_sequence_input_layer = graph['env_sequence_input_layer']
    profile_input_layer = graph['profile_input_layer']
    target_label = graph['target_label']
    accuracy = graph['accuracy']
    tfauc = graph['auc']
    softmax = graph['softmax']
    logits = graph['logits']
    attention_mask_input = graph['attention_mask_input']

    saver = tf.train.Saver()
    model_load_path = '{}-{}'.format(model_load_path, int(best_model_step))
    print('-- loading model from {} --'.format(model_load_path))

    n_test = len(test_idxes)

    # mask for attention
    profile_mask = np.full((n_test, 1), 1)
    env_mask = np.full((n_test, datas['env_sequence_features'][0].shape[0]), 1)
    lens = [len(datas['sequence_features'][idx]) for idx in test_idxes]
    sequence_mask = np.zeros((n_test, max(lens)))
    for i in range(n_test):
        sequence_mask[i, -lens[i]:] = 1
    attention_mask = np.hstack([env_mask, sequence_mask, profile_mask])

    test_feed_dict = {
        # sequence_input_layer: datas[0][test_idxes],
        sequence_input_layer: tf.keras.preprocessing.sequence.pad_sequences(
            [datas['sequence_features'][idx] for idx in test_idxes], dtype='float'),
        env_sequence_input_layer: tf.keras.preprocessing.sequence.pad_sequences(
            [datas['env_sequence_features'][idx] for idx in test_idxes], dtype='float'),
        profile_input_layer: datas['profile'][test_idxes],
        target_label: datas['labels'][test_idxes],
        attention_mask_input: attention_mask
    }

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        # print('load best model', model_load_path)
        saver.restore(sess, model_load_path)
        accuracy, softmax, tfauc, logits = sess.run(
            [accuracy, softmax, tfauc, logits], test_feed_dict)
        print('accurcy on test data', accuracy)

        y = datas['labels'][test_idxes].argmax(axis=1)
        y_pred = logits.argmax(axis=1)

        test_auc = auc(y, softmax)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        print('auc on test data', test_auc)
        return [test_auc, precision, recall, f1, accuracy]
        # print('tfauc on test data', tfauc)
        # print('softmax', softmax)


def main(params, run_from_new, model_output_path, model_load_path, data, pnratio, run_train=True, run_test=True,
         step_to_load=None, train_test_ratio=0.6, train_validate_ratio=0.6):
    from os import system
    cmd = 'rm {}* -rf'.format(model_output_path)
    print('going to execute:', cmd)
    system(cmd)

    datas_dict = prepare_data(ratio=pnratio, base_path=data)
    sequence_features, env_sequence_features, profile_features, labels = datas_dict[
                                                                             'sequence_features'], datas_dict[
                                                                             'env_sequence_features'], datas_dict[
                                                                             'profile'], datas_dict['labels']
    train_validate_idxes, test_idxes = get_train_test_split_indicies(
        train_test_ratio, labels)
    train_idxes, validate_idxes = get_train_test_split_indicies(
        train_validate_ratio, np.copy(labels[train_validate_idxes]))
    train_idxes = train_validate_idxes[train_idxes]
    validate_idxes = train_validate_idxes[validate_idxes]

    params['N_SEQUENCE_FEATUREN'] = len(sequence_features[0][0])
    params['N_ENV_SEQUENCE_FEATUREN'] = len(env_sequence_features[0][0])
    params['N_PROFILE_FEATUREN'] = profile_features.shape[1]

    graph = build_graph(params)

    if run_train:
        train(graph, run_from_new, model_output_path, model_load_path,
              params, datas_dict, train_idxes, validate_idxes)

    if run_test:
        return test(params, datas_dict, test_idxes, model_load_path, step_to_load)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from os import makedirs
    from os.path import exists, dirname, basename, join
    from myutil import file_name
    from random import shuffle
    from os import system

    parser = ArgumentParser()
    parser.add_argument('-a', '--action', type=str, default='data')
    parser.add_argument('-l', '--logdir', type=str,
                        default='logs/{}'.format(file_name(__file__)))
    parser.add_argument('-m', '--modelpath', type=str,
                        default='saved_tf_models/{}/model'.format(file_name(__file__)))
    parser.add_argument('-r', '--ratio', type=float, default=0.8)
    parser.add_argument('-s', '--step', type=int, default=400)
    parser.add_argument('-i', '--modelidx', type=int, default=None)
    parser.add_argument('-n', '--nrepeat', type=int, default=10)
    parser.add_argument('-d', '--data', type=str, default='../data/temp_output/nn_data/')
    parser.add_argument('-p', '--pnratio', type=float, default=0.8)
    parser.add_argument('-b', '--batchsize', type=int, default=128)

    args = parser.parse_args()

    if args.action == 'run':

        modeldir = dirname(args.modelpath)

        if not exists(modeldir):
            makedirs(modeldir)

        if not exists(args.logdir):
            makedirs(args.logdir)

        model_path = args.modelpath + '_run'

        params = {
            'N_CLASSES': 2,
            'LEARNING_RATE': 0.005,
            'TRAIN_STEPS': args.step,
            'RECORD_EVERY_N_ITERATIONS': 10,
            'VALIDATE_EVERY_N_ITERATIONS': 1,
            'BATCH_SIZE': args.batchsize,
            'SEQ_DENSE_ACT_FUNC': tf.nn.relu,
            'SEQ_DENSE_INITIALIZER': tf.keras.initializers.he_normal,
            'NUM_OF_RNN_CELLS': 1,
            'BI_DIRECTION_RNN': False,
            'LOG_DIR': args.logdir,
            'INITIALIZER': tf.variance_scaling_initializer
        }

        range_to_search = OrderedDict({
            'DIM_OF_ENV_RNN_OUTPUT': [5, 10, 20],
            'DIM_OF_TURNOVER_RNN_OUTPUT': [50, 70, 90, 120],
            'SEQ_DROPOUT_PROB': [0.1, 0.3, 0.5],
            'SEQ_DENSE_DIM': [10, 15, 25, 35],
            'PROFILE_DENSE_DIM': [5, 10, 20, 40, 80],
            'ATTENTION_SIZE': [10, 20, 30, 40, 50]
        })

        gridsearch_d = pd.DataFrame(
            columns=list(range_to_search.keys()) + ['test_auc', 'test_precision', 'test_recall', 'test_f1',
                                                    'test_accuracy'])

        idxes_grid = list(product(*[range(len(l))
                                    for l in range_to_search.values()]))
        shuffle(idxes_grid)

        n_total = len(idxes_grid)

        start = time()

        for i, idxes in enumerate(idxes_grid):
            params.update({k: range_to_search[k][v] for k, v in zip(
                range_to_search.keys(), idxes)})

            scores = main(params, True, model_output_path=model_path,
                          model_load_path=model_path, data=args.data, pnratio=args.pnratio, run_train=True,
                          run_test=True,
                          train_test_ratio=args.ratio, train_validate_ratio=args.ratio)

            gridsearch_d.loc[len(gridsearch_d)] = [range_to_search[k][
                                                       v] for k, v in zip(range_to_search.keys(), idxes)] + scores

            gridsearch_d.to_csv('{}.gridsearch.csv'.format(
                basename(__file__)), index=False)

            time_estimator(n_total, i + 1, start)

    else:
        prepare_data(True, None, args.data)
