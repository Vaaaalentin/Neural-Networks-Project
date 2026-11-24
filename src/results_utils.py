from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sports_10"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def callbacks_paths(model_name):
    return MODELS_DIR / f"{model_name}_best.keras", RESULTS_DIR / f"{model_name}_history.csv"


def save_results(model_name, model, test_ds, class_names, params):
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    preds = pd.DataFrame(probs, columns=class_names)
    preds.insert(0, "true_label", [class_names[i] for i in y_true])
    preds.insert(1, "pred_label", [class_names[i] for i in y_pred])
    preds.insert(2, "confidence", probs.max(axis=1))
    preds.to_csv(RESULTS_DIR / f"{model_name}_predictions.csv", index=False)

    report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                    target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULTS_DIR / f"{model_name}_report.csv")

    hist = pd.read_csv(RESULTS_DIR / f"{model_name}_history.csv")
    best = hist["val_loss"].idxmin()
    run = {
        "model": model_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "params": int(model.count_params()),
        "epochs_trained": len(hist),
        "best_epoch": int(best) + 1,
        "train_acc": round(hist.loc[best, "accuracy"], 4),
        "val_acc": round(hist.loc[best, "val_accuracy"], 4),
        "val_loss": round(hist.loc[best, "val_loss"], 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        **{f"hp_{k}": v for k, v in params.items()},
    }
    runs_path = RESULTS_DIR / "runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    pd.concat([runs, pd.DataFrame([run])], ignore_index=True).to_csv(runs_path, index=False)

    print(classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, zero_division=0))
    return y_true, y_pred, probs, test_acc
