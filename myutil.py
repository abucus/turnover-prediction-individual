def time_estimator(total, processed, start):
    from datetime import timedelta
    from time import time
    time_spent = timedelta(seconds=time() - start)
    time_remaining = time_spent / processed * (total - processed)
    print('=== remaining time {} processed {}/{}==='.format(time_remaining, processed, total))


def auc(y_true, pred, pos_label=1):
    from sklearn import metrics as m
    fpr, tpr, thresholds = m.roc_curve(y_true, pred, pos_label=pos_label)
    return m.auc(fpr, tpr)


def file_name(x):
    from os.path import basename
    x = basename(x)
    return ''.join(x.split('.')[:-1])


if __name__ == '__main__':
    # print(auc([1, 1, 2, 2], [0.1, 0.4, 0.35, 0.8],2))
    print(__file__)
    print(file_name(__file__))
