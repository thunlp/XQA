
import ujson as json
import unicodedata
from os.path import join
from typing import List

from docqa.triviaqa.trivia_qa_eval import normalize_answer as triviaqa_normalize_answer


class XQADoc(object):
    __slots__ = ["title", "answer_spans"]

    def __init__(self, title):
        self.title = title
        self.answer_spans = None

    @property
    def doc_id(self):
        return self.title

    def __repr__(self) -> str:
        return "XQADoc(%s)" % self.title


class XQAQuestion(object):
    __slots__ = ["question", "question_id", "answer", "docs"]

    def __init__(self, question, question_id, answer, docs):
        self.question = question
        self.question_id = question_id
        self.answer = answer
        self.docs = docs

    @property
    def all_docs(self):
        return self.docs

    def to_compressed_json(self):
        return [
            self.question,
            self.question_id,
            [getattr(self.answer, x) for x in self.answer.__slots__],
            [[getattr(doc, x) for x in doc.__slots__] for doc in self.docs]
        ]

    @staticmethod
    def from_compressed_json(text):
        question, quid, answer, docs = json.loads(text)
        answer = XQAAnswer(answer)
        xqa_docs = []
        for i, doc in enumerate(docs):
            xqa_docs.append(XQADoc(doc))
        return TriviaQaQuestion(question, quid, answer, xqa_docs)


class XQAAnswer(object):
    __slots__ = ["answers", "normalized_aliases"]

    def __init__(self, answers):
        self.answers = answers
        self.normalized_aliases = [triviaqa_normalize_answer(x) for x in self.answers]
        # self.normalized_aliases = [triviaqa_normalize_answer(x) if not '(' in x else triviaqa_normalize_answer(x[:x.find('(')].strip()) for x in self.answers]

    @property
    def all_answers(self):
        return self.normalized_aliases

    def __repr__(self) -> str:
        return self.answers[0]


def iter_question(filename, file_map):
    with open(filename, "r") as f:
        for line in f:
            item = json.loads(line.strip())

            question = item['question']
            question_id = item['question_id']
            answer = XQAAnswer([ans.strip() for ans in item['answers']])

            docs = []
            for (subdir, filename) in item['docs']:
                title = filename[:filename.rfind('.')] if filename.rfind('.') != -1 else filename
                filepath = join(subdir, title)
                file_map[title] = filepath
                docs.append(XQADoc(title))

            yield XQAQuestion(question, question_id, answer, docs)

