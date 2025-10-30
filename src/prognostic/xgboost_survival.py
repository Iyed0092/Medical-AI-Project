import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import json

def generate_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        'age': rng.randint(30,80,size=n),
        'bp': rng.normal(120,15,size=n),
        'chol': rng.normal(200,30,size=n),
        'sex': rng.binomial(1,0.5,n)
    })
    base = 0.01
    lin = 0.03*X['age'] + 0.02*X['bp'] - 0.01*X['chol'] + 0.2*X['sex']
    times = -np.log(rng.uniform(size=n)) / (base*np.exp(lin))
    events = rng.binomial(1,0.7,size=n)
    X['time'] = times
    X['event'] = events
    return X

def main(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = generate_data(n=args.n_samples, seed=args.seed)
    train, test = train_test_split(df, test_size=0.3, random_state=args.seed)
    dtrain = xgb.DMatrix(train[['age','bp','chol','sex']], label=train['time'])
    params = {'objective':'survival:cox','eval_metric':'cox-nloglik','eta':0.1}
    model = xgb.train(params, dtrain, num_boost_round=50)
    dtest = xgb.DMatrix(test[['age','bp','chol','sex']])
    preds = model.predict(dtest)
    try:
        auc = roc_auc_score(test['event'], preds)
    except Exception:
        auc = float('nan')
    results = {"xgb_auc": float(auc)}
    with open(out / "results.json","w") as f:
        json.dump(results,f,indent=2)
    model.save_model(str(out / "xgb_model.json"))
    test.to_csv(out / "test_data.csv", index=False)
    print("Evaluation results:", results)
    print("Saved model and test data to", out.resolve())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    from pathlib import Path
    parser.add_argument("--output_dir", type=Path, default="./experiments/xgb_run1")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
