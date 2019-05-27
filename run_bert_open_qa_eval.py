
import collections
import json
import math
import os
import random
from bert import modeling
from bert import tokenization
import optimization
import six
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "model_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "predict_file", None,
    """json file path for predictions.
       E.g., dev_output.json or test_output.json""")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("predict_batch_size", 2,
                     "Total batch size for predictions.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "max_answer_length", 10,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")


class OpenQATestExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self,
               qid,
               docid,
               question_text,
               doc_tokens,
               orig_answer_text=None):
    self.qid = qid
    self.docid = docid
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "id: %s" % (self.qid)
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    s += ", answer_text: %s" % (self.orig_answer_text)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               tokens,
               input_ids,
               input_mask,
               segment_ids):
    self.unique_id = unique_id
    self.example_index = example_index
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids


def read_open_qa_test_examples(inputfile):
  """Read a json file from DocumentQA into a list of OpenQAExample."""
  examples = []
  with open(inputfile, "r") as fin:
    for line in fin:
      item = json.loads(line.strip())
      qid = item["question_id"]
      docid = item["doc_id"]
      question_text = " ".join(item["question"]).replace("< Query >", "%q")

      doc_tokens = item["context"]
      orig_answer_text = item["answer_text"]
      example = OpenQATestExample(
          qid, docid, question_text, doc_tokens, orig_answer_text)
      examples.append(example)
  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 max_query_length, output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  c1, c2 = 0, 0 

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]
      c1 += 1

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    if example_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (unique_id))
      tf.logging.info("answer: %s" % (example.orig_answer_text))
    elif example_index % 100 == 0:
      tf.logging.info("example_index: %s" % (example_index))

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      token = token.replace("%%DOCUMENT%%", "%d")
      token = token.replace("%%PARAGRAPH%%", "%p")
      token = token.replace("%%PARAGRAPH_GROUP%%", "%g")
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in query_tokens:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in all_doc_tokens:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Truncate over long sequence
    if len(input_ids) > max_seq_length:
      input_ids = input_ids[:max_seq_length]
      input_mask = input_mask[:max_seq_length]
      segment_ids = segment_ids[:max_seq_length]
      c2 += 1

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if example_index < 20:
      tf.logging.info("#%d" % i)
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info(
          "input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    feature = InputFeatures(
        unique_id=unique_id,
        example_index=example_index,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    # Run callback
    output_fn(feature)

    unique_id += 1

  tf.logging.info("Num of overlong querys: %d" % c1)
  tf.logging.info("Num of overlong documents : %d" % c2)


def create_predict_model(bert_config, input_ids, input_mask,
                 segment_ids, use_one_hot_embeddings):
  """Creates a classification model."""
  all_logits = []
  input_ids_shape = modeling.get_shape_list(input_ids, expected_rank=2)
  batch_size = input_ids_shape[0]
  seq_length = input_ids_shape[1]

  model = modeling.BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      scope="bert")
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/open_qa/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  output_bias = tf.get_variable(
      "cls/open_qa/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])
  unstacked_logits = tf.unstack(logits, axis=0)
  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def model_fn_builder(bert_config, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    (start_logits, end_logits) = create_predict_model(
        bert_config=bert_config,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    predictions = {
        "unique_ids": unique_ids,
        "start_logits": start_logits,
        "end_logits": end_logits,
    }
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename):
    self.filename = filename
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes

def write_predictions(all_examples, all_features, all_results,
                      max_answer_length, do_lower_case, question_prediction_file,
                      paragraph_prediction_file):
  """Write final predictions to the json file."""
  tf.logging.info("Writing quesiton predictions to: %s" % (question_prediction_file))
  tf.logging.info("Writing paragraph predictions to: %s" % (paragraph_prediction_file))
  n_best_size = 30

  example_index_to_features = {} 
  for feature in all_features:
    example_index_to_features[feature.example_index] = feature

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimParagraphPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimParagreaphPrediction",
      ["start_index", "end_index", "start_logit", "end_logit"])

  _ParagraphPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "ParagreaphPrediction",
      ["qid", "docid", "pred_text", "start_index", "end_index", "start_logit", "end_logit"])

  _PrelimQuestionPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimParagreaphPrediction",
      ["example_index", "start_index", "end_index", "start_logit", "end_logit", "pred_text"])

  prelim_question_predictions = collections.defaultdict(list)
  paragraph_predictions = [] 
  for (example_index, example) in enumerate(all_examples):
    feature = example_index_to_features[example_index]
    prelim_paragraph_predictions = []
    result = unique_id_to_result[feature.unique_id]

    start_indexes = _get_best_indexes(result.start_logits, n_best_size)
    end_indexes = _get_best_indexes(result.end_logits, n_best_size)
    for start_index in start_indexes:
      for end_index in end_indexes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= len(feature.tokens):
          continue
        if end_index >= len(feature.tokens):
          continue
        if end_index < start_index:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        prelim_paragraph_predictions.append(
            _PrelimParagraphPrediction(
                start_index=start_index,
                end_index=end_index,
                start_logit=result.start_logits[start_index],
                end_logit=result.end_logits[end_index]))

    prelim_paragraph_predictions = sorted(
        prelim_paragraph_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    if len(prelim_paragraph_predictions) > 0:
        best_pred = prelim_paragraph_predictions[0]
        tok_tokens = feature.tokens[best_pred.start_index:(best_pred.end_index + 1)]
        tok_text = " ".join(tok_tokens)
    else:
        best_pred = _PrelimParagraphPrediction(
            start_index=0, end_index=0, start_logit=-float("inf"), end_logit=-float("inf"))
        tok_text = ""

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    final_text = " ".join(tok_text.split())

    paragraph_predictions.append(
        _ParagraphPrediction(
            qid=example.qid,
            docid=example.docid,
            pred_text=final_text,
            start_index=best_pred.start_index,
            end_index=best_pred.end_index,
            start_logit=best_pred.start_logit,
            end_logit=best_pred.end_logit))

    prelim_question_predictions[example.qid].append(
        _PrelimQuestionPrediction(
            example_index=example_index,
            pred_text=final_text,
            start_index=best_pred.start_index,
            end_index=best_pred.end_index,
            start_logit=best_pred.start_logit,
            end_logit=best_pred.end_logit))

  question_predictions = collections.OrderedDict()
  for (qid, question_pred) in prelim_question_predictions.items():
    question_pred = sorted(
        question_pred,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)
    question_predictions[qid] = question_pred[0].pred_text 

  with tf.gfile.GFile(question_prediction_file, "w") as writer:
    writer.write(json.dumps(question_predictions, indent=4, ensure_ascii=False) + "\n")

  with tf.gfile.GFile(paragraph_prediction_file, "w") as writer:
    for para_pred in paragraph_predictions:
      writer.write(json.dumps(para_pred, ensure_ascii=False) + "\n")


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.predict_file:
    raise ValueError(
        "`predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.model_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=os.path.join(FLAGS.model_dir, "best_model"),
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.predict_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  filename = os.path.basename(FLAGS.predict_file)
  eval_set = filename.split("_")[0]
  eval_lang = "-".join(filename.split("_")[2:-1])
  n_para = filename.split("_")[-1].split(".")[0]

  eval_examples = read_open_qa_test_examples(
      inputfile=FLAGS.predict_file)

  eval_tf_record_name = "%s-%s-%s.tf_record" % (eval_set, eval_lang, n_para)
  eval_writer = FeatureWriter(
      filename=os.path.join(FLAGS.model_dir, eval_tf_record_name))
  eval_features = []

  def append_feature(feature):
    eval_features.append(feature)
    eval_writer.process_feature(feature)

  convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      max_query_length=FLAGS.max_query_length,
      output_fn=append_feature)
  eval_writer.close()

  tf.logging.info("***** Running predictions *****")
  tf.logging.info("  Num orig examples = %d", len(eval_examples))
  tf.logging.info("  Num split examples = %d", len(eval_features))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  all_results = []

  predict_input_fn = input_fn_builder(
      input_file=eval_writer.filename,
      seq_length=FLAGS.max_seq_length,
      drop_remainder=False)

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_results = []
  for result in estimator.predict(
      predict_input_fn, yield_single_examples=True):
    if len(all_results) % 1000 == 0:
      tf.logging.info("Processing example: %d" % (len(all_results)))
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    all_results.append(
        RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits))

  question_prediction_file = "%s-question-%s-%s-output.txt" % (eval_set, eval_lang, n_para)
  paragraph_prediction_file = "%s-paragraph-%s-%s-output.txt" % (eval_set, eval_lang, n_para)
  question_prediction_file = os.path.join(FLAGS.model_dir, question_prediction_file)
  paragraph_prediction_file = os.path.join(FLAGS.model_dir, paragraph_prediction_file)

  write_predictions(eval_examples, eval_features, all_results,
                    FLAGS.max_answer_length, FLAGS.do_lower_case,
                    question_prediction_file, paragraph_prediction_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("model_dir")
  tf.app.run()
