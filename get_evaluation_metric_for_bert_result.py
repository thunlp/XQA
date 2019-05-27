
import argparse
import json

from docqa.triviaqa.trivia_qa_eval import exact_match_score
from docqa.triviaqa.trivia_qa_eval import f1_score
from docqa.triviaqa.trivia_qa_eval import metric_max_over_ground_truths

from bert.tokenization import BasicTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True) 
    parser.add_argument('--prediction_file', required=True) 
    args = parser.parse_args()
    input_file = args.input_file 
    prediction_file = args.prediction_file 

    ground_truths = {}
    tokenizer = BasicTokenizer()
    with open(input_file, "r") as fin:
      for line in fin:
        item = json.loads(line.strip())
        ground_truths[item["question_id"]] = [" ".join(tokenizer.tokenize(ans)) for ans in item["answer_text"]]

    predictions = json.load(open(prediction_file, "r"))
    f1 = []
    em = []
    for (qid, pred_text) in predictions.items():
      f1.append(metric_max_over_ground_truths(f1_score, pred_text, ground_truths[qid]))
      em.append(metric_max_over_ground_truths(exact_match_score, pred_text, ground_truths[qid]))

    import numpy as np
    print("F1:", np.mean(f1))
    print("EM:", np.mean(em))

