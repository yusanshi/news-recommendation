# News Recommendation

The repository currently includes the following models.

**Models in published papers**

| Model     | Full name                                                                 | Paper                                              |
| --------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| NRMS      | Neural News Recommendation with Multi-Head Self-Attention                 | https://www.aclweb.org/anthology/D19-1671/         |
| NAML      | Neural News Recommendation with Attentive Multi-View Learning             | https://arxiv.org/abs/1907.05576                   |
| LSTUR     | Neural News Recommendation with Long- and Short-term User Representations | https://www.aclweb.org/anthology/P19-1033.pdf      |
| DKN       | Deep Knowledge-Aware Network for News Recommendation                      | https://dl.acm.org/doi/abs/10.1145/3178876.3186175 |
| Hi-Fi Ark | Deep User Representation via High-Fidelity Archive Network                | https://www.ijcai.org/Proceedings/2019/424         |
| TANR      | Neural News Recommendation with Topic-Aware News Representation           | https://www.aclweb.org/anthology/P19-1110.pdf      |

**Experimental models**

| Model | Description |
| ----- | ----------- |
| Exp1  |             |
| Exp2  | RoBERTa     |



## Get started

Basic setup.

```bash
git clone https://github.com/yusanshi/NewsRecommendation
cd NewsRecommendation
pip3 install -r requirements.txt
```

Download and preprocess the data.

```bash
mkdir data && cd data
# Download GloVe pre-trained word embedding
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Download MIND dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
unzip MINDlarge_train.zip -d train
unzip MINDlarge_dev.zip -d val
unzip MINDlarge_test.zip -d test
rm MINDlarge_*.zip

# Preprocess data into appropriate format
cd ..
python3 src/data_preprocess.py
# Remember you shoud modify `num_*` in `src/config.py` by the output of `src/data_preprocess.py`
```

Modify `src/config.py` to select target model. The configuration file is organized into general part (which is applied to all models) and model-specific part (that some models not have).

```bash
vim src/config.py
```

Run.

```bash
# Train and save checkpoint into `checkpoint/{model_name}/` directory
python3 src/train.py
# Load latest checkpoint and evaluate on the test set
# This will also generate prediction file `data/test/prediction.txt`
python3 src/evaluate.py

# or

chmod +x run.sh
./run.sh
```

You can visualize metrics with TensorBoard.

```bash
tensorboard --logdir=runs

# or
tensorboard --logdir=runs/{model_name}
# for a specific model
```

> Tip: by adding `REMARK` environment variable, you can make the runs name in TensorBoard more meaningful. For example, `REMARK=num-filters-300-window-size-5 python3 src/train.py`.

## Results

| Model     | AUC  | nMRR | nDCG@5 | nDCG@10 | Remark |
| --------- | ---- | ---- | ------ | ------- | ------ |
| NRMS      |      |      |        |         |        |
| NAML      |      |      |        |         |        |
| LSTUR     |      |      |        |         |        |
| DKN       |      |      |        |         |        |
| Hi-Fi Ark |      |      |        |         |        |
| TANR      |      |      |        |         |        |

Checkpoints: <https://drive.google.com/open?id=TODO>

You can verify the results by simply downloading them and running `MODEL_NAME=XXXX python3 src/evaluate.py`.

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
