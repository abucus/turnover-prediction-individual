def get_train_test_split_indicies(percent_for_train, labels, random_state=None):
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    train_test_spliter = StratifiedShuffleSplit(
        n_splits=1, train_size=percent_for_train, test_size=1-percent_for_train, random_state=random_state)

    for train_indicies, test_indicies in train_test_spliter.split(np.zeros(len(labels)), labels):
        return train_indicies, test_indicies


def get_batch(datas, train_indicies, step, batch_size):
    '''
    datas is a list of global_features, sequence_features, labels
    '''
    n = len(train_indicies)
    batch_indicies = [((step-1)*batch_size+i) % n for i in range(batch_size)]
    train_indicies_for_batch = train_indicies[batch_indicies]

    dl = []
    for data in datas:
        if isinstance(data, list):
            dl.append([data[idx] for idx in train_indicies_for_batch])
        else:
            dl.append(data[train_indicies_for_batch])
    return dl


def get_batch_dict(datas, train_indicies, step, batch_size):
    '''
    datas is a dict of data
    '''
    n = len(train_indicies)
    batch_indicies = [((step-1)*batch_size+i) % n for i in range(batch_size)]
    train_indicies_for_batch = train_indicies[batch_indicies]

    batch_dict = dict()
    for k,v in datas.items():
        if isinstance(v, list):
            batch_dict[k] = [v[idx] for idx in train_indicies_for_batch]
        else:
            batch_dict[k] =v[train_indicies_for_batch]

    return batch_dict

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    labels = np.random.randint(0,2,100)
    # train_idxes, test_idxes = get_train_test_split_indicies(0.8, labels)
    # print(len(train_idxes), train_idxes)
    # print(len(test_idxes), test_idxes)

    train_validate_idxes, test_idxes = get_train_test_split_indicies(
        0.6, labels)
    train_idxes, validate_idxes = get_train_test_split_indicies(
        0.6, np.copy(labels[train_validate_idxes]))
    train_idxes = train_validate_idxes[train_idxes]
    validate_idxes = train_validate_idxes[validate_idxes]

    print(len(train_idxes))
    print(len(test_idxes))
    print(len(validate_idxes))

    print(set(train_idxes)&set(test_idxes))
    print(set(train_idxes)&set(validate_idxes))
    print(set(validate_idxes)&set(test_idxes))

    print('train:', pd.Series(labels[train_idxes]).value_counts())
    print('validate:', pd.Series(labels[validate_idxes]).value_counts())
    print('test:', pd.Series(labels[test_idxes]).value_counts())

    print(train_idxes)
    print(validate_idxes)
    print(test_idxes)
