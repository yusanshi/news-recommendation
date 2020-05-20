# News Recommendation System

Currently included model:

| Model | Full name                                                    | Paper                                              |
| ----- | ------------------------------------------------------------ | -------------------------------------------------- |
| NRMS  | Neural News Recommendation with Multi-Head Self-Attention    | https://www.aclweb.org/anthology/D19-1671/         |
| NAML  | Neural News Recommendation with Attentive Multi-View Learning | https://arxiv.org/abs/1907.05576                   |
| LSTUR | Neural News Recommendation with Long- and Short-term User Representations | https://www.aclweb.org/anthology/P19-1033.pdf      |
| DKN   | Deep Knowledge-Aware Network for News Recommendation         | https://dl.acm.org/doi/abs/10.1145/3178876.3186175 |

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
wget https://nlp.stanford.edu/data/glove.6B.zip
sudo apt install unzip
unzip glove.6B.zip -d glove
rm glove.6B.zip

# Download MIND-small dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d test
rm MINDsmall_*.zip

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
python3 src/evaluate.py

# or

chmod +x run.sh
./run.sh
```

You can visualize the training loss and accuracy with TensorBoard.

```bash
tensorboard --logdir=runs
```

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
