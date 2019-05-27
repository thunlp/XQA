import argparse
import json
import pickle
import unicodedata
from itertools import islice
from os import mkdir
from os.path import join, exists
from typing import List, Optional, Dict

from config import CORPUS_DIR, CORPUS_NAME_TO_PATH
from answer_detection import compute_answer_spans_par, FastNormalizedAnswerDetector
from evidence_corpus import XQAEvidenceCorpusTxt, ChineseTokenizer
from read_data import XQAQuestion, iter_question
from docqa.configurable import Configurable
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.utils import ResourceLoader

"""
Build span-level training data from the raw XQA inputs, in particular load the questions
from the json file and annotates each question/doc with the places the question answer's occur 
within the document, and save the results in our format. Assumes the evidence corpus has 
already been preprocessed

Modified from docqa/triviaqa/build_span_corpus.py
"""


def build_dataset(corpus_name: str, tokenizer, train_files: Dict[str, str],
                  answer_detector, n_process: int, prune_unmapped_docs=True,
                  sample=None):
    out_dir = join(CORPUS_DIR, corpus_name, corpus_name)
    if not exists(out_dir):
        mkdir(out_dir)

    file_map = {}  # maps document_id -> filename

    for name, filename in train_files.items():
        print("Loading %s questions" % name)
        if sample is None:
            questions = list(iter_question(filename, file_map))
        else:
            if isinstance(sample,  int):
                questions = list(islice(iter_question(filename, file_map), sample))
            elif isinstance(sample, dict):
                questions = list(islice(iter_question(filename, file_map), sample[name]))
            else:
                raise ValueError()

        if prune_unmapped_docs:
            for q in questions:
                q.docs = [x for x in q.docs if x.doc_id in file_map]

        print("Adding answers for %s question" % name)
        corpus = XQAEvidenceCorpusTxt(corpus_name, file_map)

        questions = compute_answer_spans_par(questions, corpus, tokenizer, answer_detector, n_process)
        new_questions = []
        for q in questions:  # Sanity check, we should have answers for everything (even if of size 0)
            if q.answer is None:
                continue
            check = True
            for doc in q.docs:
                if doc.doc_id in file_map:
                    if doc.answer_spans is None:
                        check = False
                        # raise RuntimeError()
            if check:
                new_questions.append(q)

        print("Saving %s %d question" % (name, len(new_questions)))
        with open(join(out_dir, name + ".pkl"), "wb") as f:
            pickle.dump(new_questions, f)

    print("Dumping file mapping")
    with open(join(out_dir, "file_map.json"), "w") as f:
        json.dump(file_map, f)

    print("Complete")


class XQASpanCorpus(Configurable):
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.dir = join(CORPUS_DIR, corpus_name, corpus_name)
        with open(join(self.dir, "file_map.json"), "r") as f:
            file_map = json.load(f)
        for k, v in file_map.items():
            file_map[k] = unicodedata.normalize("NFD", v)
        self.evidence = XQAEvidenceCorpusTxt(corpus_name, file_map)

    def get_train(self) -> List[XQAQuestion]:
        if exists(join(self.dir, "train.pkl")):
            with open(join(self.dir, "train.pkl"), "rb") as f:
                return pickle.load(f)
        else:
            return []

    def get_dev(self) -> List[XQAQuestion]:
        with open(join(self.dir, "dev.pkl"), "rb") as f:
            return pickle.load(f)

    def get_test(self) -> List[XQAQuestion]:
        with open(join(self.dir, "test.pkl"), "rb") as f:
            return pickle.load(f)

    """
    def get_verified(self) -> Optional[List[XQAQuestion]]:
        verified_dir = join(self.dir, "verified.pkl")
        if not exists(verified_dir):
            return None
        with open(verified_dir, "rb") as f:
            return pickle.load(f)
    """

    def get_resource_loader(self):
        return ResourceLoader()

    @property
    def name(self):
        return self.corpus_name


class XQADataset(XQASpanCorpus):
    def __init__(self, corpus_name):
        super().__init__(corpus_name)


def build_xqa_corpus(corpus_name, n_processes):
    if corpus_name.startswith("en"):
        files_dict = dict(
            train=join(CORPUS_NAME_TO_PATH[corpus_name], "qa", "train.json"),
            dev=join(CORPUS_NAME_TO_PATH[corpus_name], "qa", "dev.json"),
            test=join(CORPUS_NAME_TO_PATH[corpus_name], "qa", "test.json")
        )
    else:
        files_dict = dict(
            dev=join(CORPUS_NAME_TO_PATH[corpus_name], "qa", "dev.json"),
            test=join(CORPUS_NAME_TO_PATH[corpus_name], "qa", "test.json")
        )
    if corpus_name == "en_trans_zh" or corpus_name == "zh_ori":
        tokenizer = ChineseTokenizer()
    else:
        tokenizer = NltkAndPunctTokenizer()
    build_dataset(corpus_name, tokenizer, files_dict,
                  FastNormalizedAnswerDetector(), n_processes)


def main():
    parser = argparse.ArgumentParser("Pre-procsess XQA data")
    parser.add_argument("corpus", choices=["en", "fr", "de", "ru", "pt", "zh", "pl", "uk", "ta",
                                           "en_trans_de", "en_trans_zh",
                                           "fr_trans_en", "de_trans_en", "ru_trans_en", "pt_trans_en",
                                           "zh_trans_en", "pl_trans_en", "uk_trans_en", "ta_trans_en"])
    parser.add_argument("-n", "--n_processes", type=int, default=1, help="Number of processes to use")
    args = parser.parse_args()
    build_xqa_corpus(args.corpus, args.n_processes)


if __name__ == "__main__":
    main()


