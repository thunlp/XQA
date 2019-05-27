
import argparse
import json
from os.path import join
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from docqa import trainer
from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, FirstN
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.dataset import FixedOrderBatcher
from docqa.eval.ranked_scores import compute_ranked_scores
from docqa.evaluator import Evaluator, Evaluation
from docqa.model_dir import ModelDir
from build_span_corpus import XQADataset
from docqa.triviaqa.read_data import normalize_wiki_filename
from docqa.triviaqa.training_data import DocumentParagraphQuestion, ExtractMultiParagraphs, \
    ExtractMultiParagraphsPerQuestion
from docqa.triviaqa.trivia_qa_eval import exact_match_score as trivia_em_score
from docqa.triviaqa.trivia_qa_eval import f1_score as trivia_f1_score
from docqa.utils import ResourceLoader, print_table
from docqa.text_preprocessor import WithIndicators

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_processes', type=int, default=1,
                        help="Number of processes to do the preprocessing (selecting paragraphs+loading context) with")
    parser.add_argument('-a', '--async', type=int, default=10)
    parser.add_argument('-t', '--tokens', type=int, default=400,
                        help="Max tokens per a paragraph")
    parser.add_argument('-n', '--n_sample', type=int, default=None, help="Number of questions to evaluate on")
    parser.add_argument('-g', '--n_paragraphs', type=int, default=15,
                        help="Number of paragraphs to run the model on")
    parser.add_argument('-f', '--filter', type=str, default=None, choices=["tfidf", "truncate", "linear"],
                        help="How to select paragraphs")
    parser.add_argument('-c', '--corpus',
                        choices=["en_dev",
                                 "en_test",
                                 "fr_dev",
                                 "fr_test",
                                 "de_dev",
                                 "de_test",
                                 "ru_dev",
                                 "ru_test",
                                 "pt_dev",
                                 "pt_test",
                                 "zh_dev",
                                 "zh_test",
                                 "pl_dev",
                                 "pl_test",
                                 "uk_dev",
                                 "uk_test",
                                 "ta_dev",
                                 "ta_test",
                                 "fr_trans_en_dev",
                                 "fr_trans_en_test",
                                 "de_trans_en_dev",
                                 "de_trans_en_test",
                                 "ru_trans_en_dev",
                                 "ru_trans_en_test",
                                 "pt_trans_en_dev",
                                 "pt_trans_en_test",
                                 "zh_trans_en_dev",
                                 "zh_trans_en_test",
                                 "pl_trans_en_dev",
                                 "pl_trans_en_test",
                                 "uk_trans_en_dev",
                                 "uk_trans_en_test",
                                 "ta_trans_en_dev",
                                 "ta_trans_en_test"],
                        required=True)
    args = parser.parse_args()

    corpus_name = args.corpus[:args.corpus.rfind("_")]
    eval_set = args.corpus[args.corpus.rfind("_")+1:]
    dataset = XQADataset(corpus_name)
    if eval_set == "dev":
        test_questions = dataset.get_dev()
    elif eval_set == "test":
        test_questions = dataset.get_test()
    else:
        raise AssertionError()

    corpus = dataset.evidence
    splitter = MergeParagraphs(args.tokens)

    per_document = args.corpus.startswith("web")  # wiki and web are both multi-document

    filter_name = args.filter
    if filter_name is None:
        # Pick default depending on the kind of data we are using
        if per_document:
            filter_name = "tfidf"
        else:
            filter_name = "linear"

    print("Selecting %d paragraphs using method \"%s\" per %s" % (
        args.n_paragraphs, filter_name, ("question-document pair" if per_document else "question")))

    if filter_name == "tfidf":
        para_filter = TopTfIdf(NltkPlusStopWords(punctuation=True), args.n_paragraphs)
    elif filter_name == "truncate":
        para_filter = FirstN(args.n_paragraphs)
    elif filter_name == "linear":
        para_filter = ShallowOpenWebRanker(args.n_paragraphs)
    else:
        raise ValueError()

    n_questions = args.n_sample
    if n_questions is not None:
        test_questions.sort(key=lambda x:x.question_id)
        np.random.RandomState(0).shuffle(test_questions)
        test_questions = test_questions[:n_questions]

    preprocessor = WithIndicators()
    print("Building question/paragraph pairs...")
    # Loads the relevant questions/documents, selects the right paragraphs, and runs the model's preprocessor
    if per_document:
        prep = ExtractMultiParagraphs(splitter, para_filter, preprocessor, require_an_answer=False)
    else:
        prep = ExtractMultiParagraphsPerQuestion(splitter, para_filter, preprocessor, require_an_answer=False)
    prepped_data = preprocess_par(test_questions, corpus, prep, args.n_processes, 1000)

    data = []
    for q in prepped_data.data:
        for i, p in enumerate(q.paragraphs):
            if q.answer_text is None:
                ans = None
            else:
                ans = TokenSpans(q.answer_text, p.answer_spans)
            data.append(DocumentParagraphQuestion(q.question_id, p.doc_id,
                                                 (p.start, p.end), q.question, p.text,
                                                  ans, i))

    # Reverse so our first batch will be the largest (so OOMs happen early)
    questions = sorted(data, key=lambda x: (x.n_context_words, len(x.question)), reverse=True)

    # dump eval data for bert
    import pickle
    pickle.dump(questions, open("%s_%d.pkl" % (args.corpus, args.n_paragraphs), "wb"))

if __name__ == "__main__":
    main()

