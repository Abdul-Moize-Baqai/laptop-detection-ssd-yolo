from pathlib import Path
import csv
import json
import argparse

EXPERIMENTS = ("custom-ssd", "custom-yolo", "pretrained-ssd", "pretrained-yolo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    records = []
    for name in EXPERIMENTS:
        path = args.results_dir / name / "metrics.json"
        if not path.exists():
            print(f"NOT YET TRAINED: {name} (missing {path})")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append({k: record.get(k) for k in
                        ("experiment", "mAP50", "precision", "recall", "true_positives", "false_positives", "false_negatives", "num_train_images", "num_test_images")})
    if not records:
        print("No trained models have produced metrics.json yet.")
        return
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print("\nCompleted experiments:")
    for r in records:
        print(f"{r['experiment']:17} AP@.50={r['mAP50']:.3f}  "
              f"Precision={r['precision']:.3f}  Recall={r['recall']:.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar([r['experiment'] for r in records], [r['mAP50'] for r in records])
    ax.set_ylim(0, 1); ax.set_ylabel("AP @ IoU 0.50")
    ax.set_title("Laptop detection: custom vs. pretrained SSD & YOLO")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(args.results_dir / "comparison.png", dpi=175)
    plt.close(fig)
    print("Saved:", args.results_dir / "comparison.csv", "and comparison.png")


if __name__ == "__main__":
    main()
