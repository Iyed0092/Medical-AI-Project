import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    with open(args.input_file) as f:
        texts = [line.strip() for line in f if line.strip()]
    model = SentenceTransformer(args.model_name)
    emb = model.encode(texts, convert_to_numpy=True)
    np.save(out_dir / "embeddings.npy", emb)
    print(f"Embeddings saved to {out_dir.resolve()}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./experiments/docs.txt")
    parser.add_argument("--output_dir", type=str, default="./experiments/embeddings")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args)
