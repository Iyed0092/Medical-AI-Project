import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

def generate_data(n=500, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({'x1': rng.normal(size=n),'x2': rng.normal(size=n)})
    T = rng.binomial(1,0.5,size=n)
    y0 = 2*X['x1'] + X['x2'] + rng.normal(scale=0.5,size=n)
    y1 = y0 + 1.5 + 0.5*X['x1']
    y = y0*(1-T) + y1*T
    return X, T, y

def main(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    X, T, Y = generate_data(n=args.n_samples, seed=args.seed)
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X,T,Y,test_size=0.3,random_state=args.seed)
    X_train_s = X_train.copy()
    X_train_s['T'] = T_train
    X_test_s = X_test.copy()
    X_test_s['T'] = 0
    rf = RandomForestRegressor(n_estimators=200, random_state=args.seed)
    rf.fit(X_train_s, Y_train)
    mu0 = rf.predict(X_test_s)
    X_test_s['T'] = 1
    mu1 = rf.predict(X_test_s)
    ite = mu1 - mu0
    mse = mean_squared_error(Y_test, mu0*(1-T_test)+mu1*T_test)
    results = {"mse": float(mse), "mean_ite": float(ite.mean())}
    with open(out / "results.json","w") as f:
        json.dump(results,f,indent=2)
    import joblib
    joblib.dump(rf, out / "s_learner_model.pkl")
    pd.DataFrame(X_test).to_csv(out / "X_test.csv", index=False)
    pd.DataFrame({"T":T_test,"Y":Y_test,"ITE":ite}).to_csv(out / "predictions.csv", index=False)
    print("Saved S-Learner model and predictions to", out.resolve())
    print("Results:", results)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    from pathlib import Path
    parser.add_argument("--output_dir", type=Path, default="./experiments/s_learner_run")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
