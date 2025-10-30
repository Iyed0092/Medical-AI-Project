import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
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
    y = np.array([(bool(e), t) for e,t in zip(events,times)], dtype=[('event','bool'),('time','float64')])
    return X, y

def main(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    X, y = generate_data(n=args.n_samples, seed=args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=5, min_samples_leaf=5, random_state=args.seed)
    rsf.fit(X_train, y_train)
    pred_surv = rsf.predict(X_test)
    c_index = concordance_index_censored(y_test['event'], y_test['time'], pred_surv)[0]
    results = {"rsf_c_index": float(c_index)}
    with open(out / "results.json","w") as f:
        json.dump(results,f,indent=2)
    import joblib
    joblib.dump(rsf, out / "rsf_model.pkl")
    X_test.to_csv(out / "test_data.csv", index=False)
    print("Evaluation results:", results)
    print("Saved model and test data to", out.resolve())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    from pathlib import Path
    parser.add_argument("--output_dir", type=Path, default="./experiments/rsf_run1")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
