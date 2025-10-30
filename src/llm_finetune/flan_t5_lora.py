import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, n_samples=50, max_len=32):
        self.inputs = [f"summarize: Patient has fever {i}" for i in range(n_samples)]
        self.targets = [f"Summary {i}" for i in range(n_samples)]
        self.enc = tokenizer(self.inputs, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        self.labels = tokenizer(self.targets, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')['input_ids']
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return {'input_ids': self.enc['input_ids'][idx], 'attention_mask': self.enc['attention_mask'][idx], 'labels': self.labels[idx]}

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
    dataset = SyntheticDataset(tokenizer, n_samples=args.n_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optim.step()
    model.save_pretrained(out_dir / "flan_t5_lora")
    tokenizer.save_pretrained(out_dir / "flan_t5_lora")
    with open(out_dir / "meta.json","w") as f:
        json.dump({"n_samples": args.n_samples, "epochs": args.epochs}, f, indent=2)
    print("Fine-tuned LoRA model saved to", out_dir.resolve())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    from pathlib import Path
    parser.add_argument("--output_dir", type=Path, default="./experiments/flan_lora")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
