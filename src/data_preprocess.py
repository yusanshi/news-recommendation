from config import model_name
import pandas as pd
import json
from tqdm import tqdm
from os import path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv


def parse_behaviors(behaviors_source, behaviors_target, user2int_path):
    print(f"Parse {behaviors_source}")
    behaviors = pd.read_table(behaviors_source,
                              header=None,
                              usecols=[0, 2, 3],
                              names=['user', 'clicked_news', 'impressions'])
    behaviors.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    with tqdm(total=len(behaviors), desc="Balancing data") as pbar:
        for row in behaviors.itertuples():
            positive = [x for x in row.impressions if x.endswith('1')]
            negative = [x for x in row.impressions if x.endswith('0')]
            if len(negative) > len(positive) * Config.negative_sampling_ratio:
                negative = random.sample(
                    negative,
                    len(positive) * Config.negative_sampling_ratio)
            behaviors.at[row.Index, 'impressions'] = positive + negative
            pbar.update(1)

    user2int = {}
    for row in behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1

    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int[row.user]

    pd.DataFrame(user2int.items(), columns=['user',
                                            'int']).to_csv(user2int_path,
                                                           sep='\t',
                                                           index=False)
    print(
        f'Please modify `num_users` in `src/config.py` into 1 + {len(user2int)}'
    )

    behaviors = behaviors.explode('impressions').reset_index(drop=True)
    behaviors['candidate_news'], behaviors[
        'clicked'] = behaviors.impressions.str.split('-').str
    behaviors.to_csv(
        behaviors_target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_news(source, target, word2int_path, category2int_path, mode):
    print(f"Parse {source}")
    news = pd.read_table(
        source,
        header=None,
        usecols=range(5),
        names=['id', 'category', 'subcategory', 'title', 'abstract'])
    news.fillna(' ', inplace=True)
    parsed_news = pd.DataFrame(
        columns=['id', 'category', 'subcategory', 'title', 'abstract'])

    if mode == 'train':
        word2int = {}
        word2freq = {}
        category2int = {}

        for row in news.itertuples(index=False):
            for w in word_tokenize(row.title.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.abstract.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        with tqdm(total=len(news),
                  desc="Parsing categories and words") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id,
                    category2int[row.category], category2int[row.subcategory],
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract
                ]
                try:
                    for i, w in enumerate(word_tokenize(row.title.lower())):
                        if w in word2int:
                            new_row[3][i] = word2int[w]
                except IndexError:
                    pass

                try:
                    for i, w in enumerate(word_tokenize(row.abstract.lower())):
                        if w in word2int:
                            new_row[4][i] = word2int[w]
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)
        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )
        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}'
        )

    elif mode == 'test':

        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        category2int = dict(pd.read_table(category2int_path).values.tolist())

        word_total = 0
        word_missed = 0

        with tqdm(total=len(news),
                  desc="Parsing categories and words") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id, category2int[row.category] if row.category
                    in category2int else 0, category2int[row.subcategory]
                    if row.subcategory in category2int else 0,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract
                ]
                try:
                    for i, w in enumerate(word_tokenize(row.title.lower())):
                        word_total += 1
                        if w in word2int:
                            new_row[3][i] = word2int[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                try:
                    for i, w in enumerate(word_tokenize(row.abstract.lower())):
                        word_total += 1
                        if w in word2int:
                            new_row[4][i] = word2int[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        print(f'Out-of-Vocabulary rate: {word_missed/word_total:.4f}')
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


def generate_embedding(source, target, word2int_path):
    # na_filter=False is needed since nan is also a valid word
    word2int = dict(
        pd.read_table(word2int_path, na_filter=False).values.tolist())
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE)
    source_embedding['vector'] = source_embedding.values.tolist()
    target_embedding = np.random.normal(size=(1 + len(word2int),
                                              Config.word_embedding_dim))
    target_embedding[0] = 0
    word_missed = 0
    with tqdm(total=len(word2int),
              desc="Generating word embedding from pretrained embedding file"
              ) as pbar:
        for k, v in word2int.items():
            if k in source_embedding.index:
                target_embedding[v] = source_embedding.loc[k].vector
            else:
                word_missed += 1

            pbar.update(1)

    print(
        f'Rate of word missed in pretrained embedding: {word_missed/len(word2int):.4f}'
    )
    np.save(target, target_embedding)


def transform2json(source, target):
    """
    Transform bahaviors file in tsv to json for later evaluation
    """
    behaviors = pd.read_table(
        source,
        header=None,
        names=['uid', 'time', 'clicked_news', 'impression'])
    f = open(target, "w")
    with tqdm(total=len(behaviors), desc="Transforming tsv to json") as pbar:
        for row in behaviors.itertuples(index=False):
            item = {}
            item['uid'] = row.uid[1:]
            item['time'] = row.time
            item['impression'] = {
                x.split('-')[0][1:]: int(x.split('-')[1])
                for x in row.impression.split()
            }
            f.write(json.dumps(item) + '\n')

            pbar.update(1)

    f.close()


if __name__ == '__main__':
    if model_name == 'NRMS':
        from config import NRMSConfig as Config
    elif model_name == 'NAML':
        from config import NAMLConfig as Config
    elif model_name == 'LSTUR':
        from config import LSTURConfig as Config
    else:
        print("Model name not included!")
        exit()

    train_dir = './data/train'
    test_dir = './data/test'

    print('Process data for training')

    print('Parse behaviors')
    parse_behaviors(path.join(train_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'behaviors_parsed.tsv'),
                    path.join(train_dir, 'user2int.tsv'))

    print('Parse news')
    parse_news(path.join(train_dir, 'news.tsv'),
               path.join(train_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               mode='train')

    print('Generate embedding')
    generate_embedding(
        f'./data/glove/glove.6B.{Config.word_embedding_dim}d.txt',
        './data/pretrained_word_embedding.npy',
        path.join(train_dir, 'word2int.tsv'))

    print('\nProcess data for evaluation')

    print('Transform test data')
    transform2json(path.join(test_dir, 'behaviors.tsv'),
                   path.join(test_dir, 'truth.json'))

    print('Parse news')
    parse_news(path.join(test_dir, 'news.tsv'),
               path.join(test_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               mode='test')
