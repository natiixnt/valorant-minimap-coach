#!/usr/bin/env python3
"""
Train the agent footstep classifier from collected samples.

Usage:
    python tools/train_classifier.py
    python tools/train_classifier.py --samples data/footsteps/ --output data/footstep_model.pkl
    python tools/train_classifier.py --evaluate   # cross-validation report
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train footstep agent classifier")
    parser.add_argument("--samples", default="data/footsteps/", help="Sample directory")
    parser.add_argument("--output", default="data/footstep_model.pkl", help="Model output path")
    parser.add_argument("--evaluate", action="store_true", help="Run cross-validation")
    args = parser.parse_args()

    samples_dir = Path(args.samples)
    if not samples_dir.exists():
        print(f"[Train] Samples directory not found: {samples_dir}")
        print(f"        Run: python tools/collect_footsteps.py --agent <name>")
        sys.exit(1)

    # Count samples
    agents = [d for d in sorted(samples_dir.iterdir()) if d.is_dir()]
    if not agents:
        print(f"[Train] No agent subdirectories found in {samples_dir}")
        sys.exit(1)

    print("[Train] Sample counts:")
    total = 0
    for a in agents:
        n = len(list(a.glob("*.npy"))) + len(list(a.glob("*.wav")))
        print(f"  {a.name:20s} {n} samples")
        total += n
    print(f"  Total: {total} samples across {len(agents)} agents")

    if total < 10:
        print("[Train] Too few samples. Collect at least 20 per agent.")
        sys.exit(1)

    from src.audio.agent_classifier import AgentClassifier
    clf = AgentClassifier()

    if args.evaluate:
        _cross_validate(samples_dir)
    else:
        clf.train(str(samples_dir))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        clf.save(args.output)
        print(f"[Train] Model saved to {args.output}")
        print(f"[Train] Restart the coach to load the new model.")


def _cross_validate(samples_dir: Path) -> None:
    """5-fold cross-validation accuracy report."""
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("[Train] pip install scikit-learn")
        return

    import numpy as np
    from src.audio.agent_classifier import AgentClassifier, extract_features

    clf_obj = AgentClassifier()
    X, y_raw = [], []
    for agent_dir in sorted(samples_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name.lower()
        for clip_path in list(agent_dir.glob("*.npy")) + list(agent_dir.glob("*.wav")):
            audio = clf_obj._load_clip(clip_path)
            if audio is None:
                continue
            X.append(extract_features(audio))
            y_raw.append(agent_name)

    if len(X) < 10:
        print("[Train] Not enough data for cross-validation.")
        return

    X_arr = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X_arr, y, cv=min(5, len(set(y_raw))), scoring="accuracy")
    print(f"[Train] Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"        Fold scores: {scores}")

    # Confusion matrix
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report
    import numpy as np

    skf = StratifiedKFold(n_splits=min(5, len(set(y_raw))), shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in skf.split(X_arr, y):
        clf.fit(X_arr[train_idx], y[train_idx])
        y_pred_all.extend(clf.predict(X_arr[test_idx]))
        y_true_all.extend(y[test_idx])

    labels = le.classes_
    print("\n[Train] Classification report:")
    print(classification_report(y_true_all, y_pred_all, target_names=labels, zero_division=0))


if __name__ == "__main__":
    main()
