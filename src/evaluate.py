# Credit
# Based on https://github.com/msnews/MIND/blob/master/evaluate.py by @yjw1029

import numpy as np
import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def score(truth_f, sub_f):
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    lts = truth_f.readlines()
    lss = sub_f.readlines()

    with tqdm(total=min(len(lts), len(lss)), desc="Evaluating") as pbar:
        for lt, ls in zip(lts, lss):
            lt = json.loads(lt)
            ls = json.loads(ls)

            assert lt['uid'] == ls['uid'] and lt['time'] == ls['time']

            y_true = []
            y_score = []

            ltsess = lt['impression']
            lfsess = ls['impression']

            for k, v in ltsess.items():
                y_true.append(v)
                y_score.append(lfsess[k])

            auc = roc_auc_score(y_true, y_score)
            mrr = mrr_score(y_true, y_score)
            ndcg5 = ndcg_score(y_true, y_score, 5)
            ndcg10 = ndcg_score(y_true, y_score, 10)

            aucs.append(auc)
            mrrs.append(mrr)
            ndcg5s.append(ndcg5)
            ndcg10s.append(ndcg10)

            pbar.update(1)

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)


if __name__ == '__main__':
    truth_file = open('./data/test/truth.json', 'r')
    submission_answer_file = open('./data/test/answer.json', 'r')
    output_file = open('./data/test/scores.txt', 'w')

    auc, mrr, ndcg5, ndcg10 = score(truth_file, submission_answer_file)
    result = f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    print(result)
    output_file.write(result)
    output_file.close()
