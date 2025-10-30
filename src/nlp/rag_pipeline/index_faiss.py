import argparse
import faiss
import numpy as np
import json
from pathlib import Path

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(args.embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "faiss_index.index"))
    np.save(out_dir / "doc_ids.npy", np.arange(len(embeddings)))
    with open(out_dir / "meta.json","w") as f:
        json.dump({"n_docs": len(embeddings)}, f, indent=2)
    print(f"FAISS index saved to {out_dir.resolve()}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="./experiments/embeddings.npy")
    parser.add_argument("--output_dir", type=str, default="./experiments/faiss_index")
    args = parser.parse_args()
    main(args)
