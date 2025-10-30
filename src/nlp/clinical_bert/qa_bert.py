import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

def main(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
    in_file = Path(args.input_file)
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(in_file) as f:
        data = json.load(f)
    answers = []
    for record in data:
        context = record.get("context","")
        question = record.get("question","")
        ans = qa_pipe(question=question, context=context)
        record["answer"] = ans.get("answer","")
        answers.append(record)
    with open(out_file,"w") as f:
        json.dump(answers,f,indent=2)
    print(f"QA results saved to {out_file.resolve()}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./experiments/qa_input.json")
    parser.add_argument("--output_file", type=str, default="./experiments/qa_output.json")
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    args = parser.parse_args()
    main(args)
