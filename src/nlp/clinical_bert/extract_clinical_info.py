import argparse
import pandas as pd
import re
from pathlib import Path
import json

def extract_info(text):
    age = re.search(r'(\d{2})\s*years?', text)
    sex = re.search(r'\b(male|female)\b', text, re.IGNORECASE)
    bp = re.search(r'blood pressure[:\s]+(\d{2,3})', text, re.IGNORECASE)
    return {
        "age": int(age.group(1)) if age else None,
        "sex": sex.group(1).lower() if sex else None,
        "blood_pressure": int(bp.group(1)) if bp else None
    }

def main(args):
    in_file = Path(args.input_file)
    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_file)
    extracted = [extract_info(t) for t in df['text']]
    pd.DataFrame(extracted).to_csv(out_file, index=False)
    with open(out_file.parent / "meta.json","w") as f:
        json.dump({"n_records": len(extracted)}, f, indent=2)
    print(f"Extracted info saved to {out_file.resolve()}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./experiments/clinical_text.csv")
    parser.add_argument("--output_file", type=str, default="./experiments/clinical_info.csv")
    args = parser.parse_args()
    main(args)
