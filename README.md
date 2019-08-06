## XQA
This repo contains data and baseline implementation for ACL 2019 paper "XQA: A Cross-lingual Open-domain Question Answering Dataset".


## Setup
### Data
The XQA dataset (questions, answers, and top-10 relevant articles) can be downloaded with the following link: [The XQA dataset](https://thunlp.s3-us-west-1.amazonaws.com/data_XQA.tar.gz). 
<!--https://thunlp.oss-cn-qingdao.aliyuncs.com/data_XQA.tar.gz-->

We also provide preprocessed wiki dumps for each language at [Wiki Dump for XQA](https://thunlp.oss-cn-qingdao.aliyuncs.com/wiki_XQA.tar.gz). If you are going to use your own retrival module, please use them as the text corpus.


### Dependencies
Our implementation bases on [DocumentQA](https://github.com/allenai/document-qa) and [BERT](https://github.com/google-research/bert) and we use them as submodules.

After you clone our repo, fetch the submodules with:
```
git submodule init
git submodule update
```

We require python >= 3.5, tensorflow, and other supporting libraries for [DocumentQA](https://github.com/allenai/document-qa) and [BERT](https://github.com/google-research/bert).

To install the dependencies for DocumentQA other than tensorflow, use

`pip install -r documentqa/requirements.txt`

The stopword corpus and punkt sentence tokenizer for nltk are needed and can be fetched with:

`python -m nltk.downloader punkt stopwords`

It should be noted that [DocumentQA](https://github.com/allenai/document-qa) and [BERT](https://github.com/google-research/bert) require different versions of Tensorflow.

To train and validate models with DocumentQA, use:

`pip install tensorflow-gpu==1.3.0`

To train and validate models with DocumentQA, use:

`pip install tensorflow-gpu==1.11.0`

The easiest way to run this code is to use:

``export PYTHONPATH=${PYTHONPATH}:`pwd`/documentqa``


### Word Vectors
The DocumentQA models use the common crawl 840 billion token GloVe word vectors from [here](https://nlp.stanford.edu/projects/glove/).
They are expected to exist in "\~/data/glove/glove.840B.300d.txt" or "\~/data/glove/glove.840B.300d.txt.gz".

For example:

```
mkdir -p ~/data
mkdir -p ~/data/glove
cd ~/data/glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

### Data Preprocessing
First, set "DATA_DIR" in config.py to path which stores XQA data.

To preprocess the data, we can run the following code for each corpus (en, de, fr, pl, pt, ru, ta, uk, zh):

```
python preprocess_data.py <corpus_name>
python evidence_corpus.py --corpus <corpus_name> --n_processes 8
python build_span_corpus.py <corpus_name> --n_processes 8
```

### Training Model with DocumentQA
After data preprocessing, use "ablate_xqa.py" to train DocumentQA models, for example:
`python ablate_xqa.py <corpus_name> shared-norm <model_dir>`


### Evaluating Model with DocumentQA
To evaluate DocumentQA models, use "document_qa_eval.py", for example:
`python document_qa_eval.py --n_processes 8 -c <corpus_name> --tokens 400 -o <question_output> -p <paragraph_output> <model_dir> --n_paragraphs 5`


### Training Model with BERT
To handle multiple paragraphs for a single question, following [Clark and Gardner](https://www.aclweb.org/anthology/P18-1078), we adopt shared-normalization as the training objective on sampling paragraphs as training object for BERT model. We use code in DocumentQA to sample paragraphs and transform the data format for BERT, for example:

```
python cache_train.py en shared-norm
python dump_preprocessed_train.py --input_file train_data.pkl --output_train_file en_train_output.json --num_epoch 10
python cache_dev.py en shared-norm
python dump_preprocessed_dev.py --input_file dev_data_en.pkl --output_dev_file en_dev_output.json
```

Then we could train BERT model, for example:

```
python run_bert_open_qa_train.py --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt --train_file=en_train_output.json --eval_file=en_dev_output.json --train_batch_size=2 --num_gpus 2 --learning_rate=3e-5 --num_train_epochs=1 --max_seq_length=512 --max_query_length=128 --output_dir=<model_dir> --do_lower_case=False
```

### Evaluating Model with BERT
To evaluate BERT model, we first generate test file, for example:
```
python cache_test.py --corpus de_test --n_paragraphs 5 --tokens 400
python dump_preprocessed_eval.py --input_file de_test_5.pkl --output_file test_output_de_5.json
```

Next, we run evaluation and get metrics (EM & F1 score), for example:
```
python run_bert_open_qa_eval.py --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt --predict_file=test_output_de_5.json --predict_batch_size=4 --max_seq_length=512 --max_query_length=128 --model_dir=<model_dir> --do_lower_case=False
python get_evaluation_metric_for_bert_result.py --input_file test_output_de_5.json --prediction_file <model_dir>/test-question-de-5-output.txt
```

### Cite
If you use the code, please cite this paper:

```
@inproceedings{liu2019xqa,
  title={{XQA}: A Cross-lingual Open-domain Question Answering Dataset},
  author={Liu, Jiahua and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of ACL 2019},
  year={2019}
}
```
