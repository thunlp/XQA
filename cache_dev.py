import argparse
from datetime import datetime

from docqa import model_dir
from docqa import trainer
from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.data_processing.qa_training_data import ContextLenBucketedKey
from docqa.dataset import ClusteredBatcher
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from docqa.scripts.ablate_triviaqa import get_model
from docqa.text_preprocessor import WithIndicators
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion
from build_span_corpus import XQADataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", choices=["en", "fr", "de", "ru", "pt", "zh", "pl", "uk", "ta"])
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm",
                                         "sigmoid", "paragraph"])
    # Note I haven't tested modes other than `shared-norm` on this corpus, so
    # some things might need adjusting
    parser.add_argument("-t", "--n_tokens", default=400, type=int,
                        help="Paragraph size")
    args = parser.parse_args()
    mode = args.mode
    corpus = args.corpus

    model = get_model(100, 140, mode, WithIndicators())

    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(args.n_tokens),
                                                ShallowOpenWebRanker(16),
                                                model.preprocessor, intern=True)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge", per_doc=False)]
    oversample = [1] * 2  # Sample the top two answer-containing paragraphs twice

    if mode == "paragraph":
        n_epochs = 120
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True),
                                          oversample,  only_answers=True)
    elif mode == "confidence" or mode == "sigmoid":
        if mode == "sigmoid":
            n_epochs = 640
        else:
            n_epochs = 160
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), oversample)
    else:
        n_epochs = 80
        test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, oversample)
        train = StratifyParagraphSetsBuilder(30, mode == "merge", True, oversample)

    data = XQADataset(corpus)

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)

    data.preprocess(1, 1000)

    # dump preprocessed dev data for bert
    data.cache_preprocess("dev_data_%s.pkl" % args.corpus)


if __name__ == "__main__":
    main()

