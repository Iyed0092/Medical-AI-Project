import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pathlib import Path
import json

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.read_index(args.index_file)
    doc_ids = np.load(args.doc_ids)
    model = SentenceTransformer(args.embed_model)
    qa_model = pipeline("question-answering", model=args.qa_model, tokenizer=args.qa_model)
    questions = []
    with open(args.questions) as f:
        questions = [q.strip() for q in f if q.strip()]
    embeddings = model.encode(questions, convert_to_numpy=True)
    results = []
    for q, emb in zip(questions, embeddings):
        D, I = index.search(emb.reshape(1,-1), args.top_k)
        doc_id = doc_ids[I[0][0]]
        context = f"Document ID {doc_id}"
        ans = qa_model(question=q, context=context)
        results.append({"question": q, "answer": ans.get("answer",""), "doc_id": int(doc_id)})
    with open(out_dir / "qa_results.json","w") as f:
        json.dump(results,f,indent=2)
    print(f"RAG QA results saved to {out_dir.resolve()}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, default="./experiments/faiss_index/faiss_index.index")
    parser.add_argument("--doc_ids", type=str, default="./experiments/faiss_index/doc_ids.npy")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--qa_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--questions", type=str, default="./experiments/questions.txt")
    parser.add_argument("--output_dir", type=str, default="./experiments/rag_qa_out")
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()
    main(args)
