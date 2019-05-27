
import argparse
import pickle
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True,
                        help="input file, e.g. eval_questions.pkl")
    parser.add_argument('--output_file', required=True,
                        help="output file, e.g. eval_output.json")
    args = parser.parse_args()

    data = pickle.load(open(args.input_file, "rb")) 
    outputs = []
    for item in data:
        current = {"question_id": "", "doc_id": "", "question": [], "context": [], "answer_text": []}
        current = {}
        current["question_id"] = item.question_id 
        current["doc_id"] = item.doc_id
        current["question"] = item.question
        current["context"] = item.context
        current["answer_text"] = item.answer.answer_text
        outputs.append(current)

    with open(args.output_file, "w") as fout:
        for output in outputs:
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

