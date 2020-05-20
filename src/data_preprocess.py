from config import model_name
import pandas as pd
import json
from tqdm import tqdm
from os import path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv


def parse_behaviors(source, target, user2int_path):
    print(f"Parse {source}")
    behaviors = pd.read_table(source,
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
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_news(source, target, category2int_path, word2int_path, entity2int_path, mode):
    print(f"Parse {source}")
    news = pd.read_table(
        source,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6],
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'entities'])
    news.entities.fillna('[]', inplace=True)
    news.fillna(' ', inplace=True)
    parsed_news = pd.DataFrame(
        columns=['id', 'category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities'])

    if mode == 'train':
        category2int = {}
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        for row in news.itertuples(index=False):
            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

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
            for e in json.loads(row.entities):
                # Count occurrence time within title and abstract
                times = len(
                    list(
                        filter(lambda x: x < len(row.title) + len(row.abstract) + 1,
                               e['OccurrenceOffsets']))) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= Config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        with tqdm(total=len(news),
                  desc="Parsing categories, words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id,
                    category2int[row.category],
                    category2int[row.subcategory],
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]

                try:
                    for i, w in enumerate(word_tokenize(row.title.lower())):
                        if w in word2int:
                            new_row[3][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[5][i] = local_entity_map[w]
                except IndexError:
                    pass

                try:
                    for i, w in enumerate(word_tokenize(row.abstract.lower())):
                        if w in word2int:
                            new_row[4][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[6][i] = local_entity_map[w]
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)

        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )

        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
    elif mode == 'test':
        category2int = dict(pd.read_table(category2int_path).values.tolist())
        # na_filter=False is needed since nan is also a valid word
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        word_total = 0
        word_missed = 0

        with tqdm(total=len(news),
                  desc="Parsing categories, words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id,
                    category2int[row.category] if row.category in category2int else 0,
                    category2int[row.subcategory] if row.subcategory in category2int else 0,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]

                try:
                    for i, w in enumerate(word_tokenize(row.title.lower())):
                        word_total += 1
                        if w in word2int:
                            new_row[3][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[5][i] = local_entity_map[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                try:
                    for i, w in enumerate(word_tokenize(row.abstract.lower())):
                        word_total += 1
                        if w in word2int:
                            new_row[4][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[6][i] = local_entity_map[w]
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


def generate_word_embedding(source, target, word2int_path):
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


def transform_entity_embedding(source, target, entity2int_path):
    """
    Args:
        source: path of embedding file
            example:
                Q100	-0.075855	-0.164252	0.128812	-0.022738	-0.127613	-0.160166	0.138481	-0.135568	0.117921	-0.003037	0.127557	0.142511	0.084117	-0.004320	-0.090240	0.009786	0.013588	0.003356	-0.066014	-0.098590	-0.088168	0.055409	-0.004417	0.118718	-0.035986	-0.010574	0.060249	0.064847	0.106534	0.015566	-0.077538	0.027226	0.040080	-0.132547	-0.015346	0.048049	-0.139377	-0.152344	-0.050292	0.022452	-0.122296	-0.026120	0.008042	-0.059975	-0.132461	-0.102174	-0.122510	0.008978	-0.011055	0.114250	-0.109533	0.012790	0.120282	0.031591	0.043915	-0.014192	-0.000558	-0.009249	-0.023576	-0.054018	-0.143273	0.131889	0.090060	0.056647	0.062646	-0.198711	-0.162954	-0.160493	-0.042409	-0.043214	-0.117995	-0.160036	0.090786	0.129228	-0.118732	-0.022712	-0.001741	0.156582	0.011148	0.027286	0.047676	0.002435	0.019395	0.140718	0.139035	-0.081709	0.034342	0.059993	-0.141031	-0.072964	-0.104429	0.084221	0.036348	-0.128924	-0.228023	-0.180280	-0.025696	-0.141512	0.037383	0.085674
        target: path of transformed embedding file in numpy format
        entity2int_path
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:,
                                                       1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector'
                                         ]].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(entity_embedding, entity2int,
                         on='entity').sort_values('int')
    # TODO in fact, some entity in entity2int cannot be found in entity_embedding
    # see https://github.com/msnews/MIND/issues/2
    entity_embedding_transformed = np.zeros(
        (len(entity2int) + 1, Config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)


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
    elif model_name == 'DKN':
        from config import DKNConfig as Config
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
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='train')

    print('Generate word embedding')
    generate_word_embedding(
        f'./data/glove/glove.6B.{Config.word_embedding_dim}d.txt',
        path.join(train_dir, 'pretrained_word_embedding.npy'),
        path.join(train_dir, 'word2int.tsv'))

    print('Transform entity embeddings')
    transform_entity_embedding(path.join(train_dir, 'entity_embedding.vec'),
                               path.join(train_dir, 'entity_embedding.npy'),
                               path.join(train_dir, 'entity2int.tsv'))

    print('\nProcess data for evaluation')

    print('Transform test data')
    transform2json(path.join(test_dir, 'behaviors.tsv'),
                   path.join(test_dir, 'truth.json'))

    print('Parse news')
    parse_news(path.join(test_dir, 'news.tsv'),
               path.join(test_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='test')
