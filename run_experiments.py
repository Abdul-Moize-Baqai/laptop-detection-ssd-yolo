from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset_utils import (
    LaptopCOCO, collate_detection, evaluate_ap50, export_ultralytics_yolo,
    load_or_create_split, save_prediction_pair, to_device,
)
from models import (
    TinyYOLO, build_custom_ssd, build_pretrained_ssd, make_yolo_targets,
    yolo_loss, yolo_resize_batch,
)


EXPERIMENTS = ("custom-ssd", "custom-yolo", "pretrained-ssd", "pretrained-yolo")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(explicit):
    return torch.device(explicit if explicit else ("cuda" if torch.cuda.is_available() else "cpu"))


def make_loaders(dataset, split, batch_size):
    train = LaptopCOCO(dataset.images_root, dataset.annotation_file, split["train"])
    test = LaptopCOCO(dataset.images_root, dataset.annotation_file, split["test"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_detection, num_workers=0)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, collate_fn=collate_detection, num_workers=0)
    return train_loader, test_loader, test


def save_learning_curve(history, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set(xlabel="Training epoch", ylabel="Mean training loss", title="Detector training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=170)
    plt.close(fig)


def evaluate_torchvision_ssd(model, test_loader, device, score_threshold=0.05):
    model.eval()
    predictions, targets = {}, {}
    with torch.inference_mode():
        for images, target_list in test_loader:
            outputs = model([im.to(device) for im in images])
            for target, out in zip(target_list, outputs):
                image_id = int(target["image_id"].item())
                targets[image_id] = target["boxes"].cpu()
                predictions[image_id] = {"boxes": out["boxes"].cpu(), "scores": out["scores"].cpu()}
    return evaluate_ap50(predictions, targets, score_threshold=score_threshold), predictions


def evaluate_custom_yolo(model, test_loader, score_threshold=0.05):
    model.eval()
    predictions, targets = {}, {}
    with torch.inference_mode():
        for images, tlist in test_loader:
            out_list = model.predict(images, score_threshold=score_threshold)
            for target, out in zip(tlist, out_list):
                image_id = int(target["image_id"].item())
                targets[image_id] = target["boxes"].cpu()
                predictions[image_id] = out
    return evaluate_ap50(predictions, targets, score_threshold=score_threshold), predictions


def build_model(experiment, args, device):
    if experiment == "custom-ssd":
        return build_custom_ssd().to(device)
    if experiment == "pretrained-ssd":
        print("Loading COCO-pretrained SSD300-VGG16 (first run downloads weights).")
        return build_pretrained_ssd(download_weights=True).to(device)
    if experiment == "custom-yolo":
        return TinyYOLO(input_size=args.image_size_yolo).to(device)
    raise ValueError(experiment)


def train_torch_model(experiment, args, dataset, split, device, out_dir):
    batch_size = args.batch_size or (1 if experiment == "pretrained-ssd" else 2 if experiment == "custom-ssd" else 4)
    train_loader, test_loader, test_set = make_loaders(dataset, split, batch_size)
    model = build_model(experiment, args, device)
    learning_rate = args.lr if args.lr is not None else (1e-4 if experiment == "pretrained-ssd" else 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        running, count = 0.0, 0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if experiment in ("custom-ssd", "pretrained-ssd"):
                losses = model([im.to(device) for im in images], [to_device(t, device) for t in targets])
                loss = sum(losses.values())
            else:
                batch, targets_resized = yolo_resize_batch(images, targets, model.input_size, device)
                logits = model(batch)
                y = make_yolo_targets(targets_resized, model.input_size, model.anchors,
                                      logits.shape[1], device)
                loss, _parts = yolo_loss(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running += float(loss.detach())
            count += 1
        mean = running / max(count, 1)
        history.append(mean)
        print(f"{experiment}: epoch {epoch + 1:02d}/{args.epochs} | mean train loss {mean:.4f}", flush=True)
    save_learning_curve(history, out_dir)
    checkpoint = {
        "model_state": model.state_dict(), "model_type": experiment,
        "epoch": args.epochs, "seed": args.seed, "split_file": str(args.results_dir / "split_ids.json"),
        "history": history,
        "image_size_yolo": args.image_size_yolo if experiment == "custom-yolo" else None,
    }
    torch.save(checkpoint, out_dir / "model_checkpoint.pth")
    if experiment == "custom-yolo":
        metrics, predictions = evaluate_custom_yolo(model, test_loader)
    else:
        metrics, predictions = evaluate_torchvision_ssd(model, test_loader, device)
    metrics.update({"experiment": experiment, "epoch_count": args.epochs,
                    "num_train_images": len(split["train"]), "num_val_images": len(split["val"]),
                    "num_test_images": len(split["test"]), "final_train_loss": history[-1],
                    "learning_rate": learning_rate, "batch_size": batch_size})
    return metrics, predictions, test_set


def train_pretrained_yolo(args, dataset, split, device, out_dir):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install Ultralytics using: python -m pip install ultralytics") from exc
    data_yaml = export_ultralytics_yolo(dataset, split, args.results_dir / "prepared_yolo")
    print("Fine-tuning pretrained", args.yolo_weights)
    yolo = YOLO(args.yolo_weights)
    if device.type == "cuda":
        device_argument = device.index if device.index is not None else 0
    else:
        device_argument = "cpu"
    batch_size = args.batch_size or 4
    yolo.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.image_size_yolo,
        batch=batch_size, seed=args.seed, workers=0,
        project=str(out_dir.parent.resolve()), name=out_dir.name, exist_ok=True,
        device=device_argument, patience=15, plots=True,
        amp=(device.type == "cuda"),
    )
    
    checkpoint = out_dir / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ultralytics did not create expected checkpoint: {checkpoint}")
    yolo = YOLO(str(checkpoint))
    predictions, targets = {}, {}
    test_set = LaptopCOCO(dataset.images_root, dataset.annotation_file, split["test"])
    for image_id in split["test"]:
        full_path = test_set.path_for(image_id)
        with Image.open(full_path) as im:
            width, height = im.size
        result = yolo.predict(source=str(full_path), conf=0.05, iou=0.45,
                              imgsz=args.image_size_yolo, max_det=100,
                              device=device_argument, verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu() if result.boxes is not None else torch.zeros(0, 4)
        scores = result.boxes.conf.detach().cpu() if result.boxes is not None else torch.zeros(0)
        predictions[image_id] = {"boxes": boxes, "scores": scores}
        targets[image_id] = test_set.target_for(image_id, (width, height))["boxes"]
    metrics = evaluate_ap50(predictions, targets)
    metrics.update({"experiment": "pretrained-yolo", "epoch_count": args.epochs,
                    "num_train_images": len(split["train"]), "num_val_images": len(split["val"]),
                    "num_test_images": len(split["test"]),
                    "pretrained_weights": args.yolo_weights, "batch_size": batch_size})
    return metrics, predictions, test_set


def smoke_test(device):
    print("Smoke test | device:", device)
    
    ssd = build_custom_ssd().to(device)
    ssd.eval()
    with torch.no_grad():
        out = ssd([torch.rand(3, 300, 300, device=device)])
    assert out[0]["boxes"].shape[-1] == 4
    print("  custom SSD    -> boxes", tuple(out[0]["boxes"].shape))
    del ssd, out
    yolo = TinyYOLO(input_size=416).to(device)
    x = torch.rand(1, 3, 416, 416, device=device)
    logits = yolo(x)
    assert tuple(logits.shape) == (1, 13, 13, 3, 6)
    gt = [{"boxes": torch.tensor([[70., 90., 250., 290.]], device=device)}]
    target = make_yolo_targets(gt, 416, yolo.anchors, 13, device)
    loss, components = yolo_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    print("  custom YOLO   -> output", tuple(logits.shape), "| train loss", round(float(loss.detach()), 4))
    yolo.eval()
    with torch.no_grad():
        out = yolo.predict([torch.rand(3, 240, 320)], score_threshold=0.90)
    assert out[0]["boxes"].shape[-1] == 4
    print("  YOLO decode   -> boxes", tuple(out[0]["boxes"].shape))
    del yolo, logits, loss, x
    
    pre = build_pretrained_ssd(download_weights=False)
    assert pre.head.classification_head.module_list[0].out_channels == 8
    print("  SSD300 head   -> 2 classes (background + laptop)")
    print("Smoke test PASSED. This does NOT imply model accuracy or completed training.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", type=Path, default=Path("laptop_all_3_batches_COCO.json"))
    p.add_argument("--images", type=Path, default=Path("raw_images"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu or cuda; autodetect if omitted")
    p.add_argument("--image-size-yolo", type=int, default=416)
    p.add_argument("--yolo-weights", default="yolo11n.pt", help="Official pretrained Ultralytics checkpoint")
    p.add_argument("--prepare", action="store_true", help="Create train/val/test split and Ultralytics YOLO files")
    p.add_argument("--report-dataset", action="store_true", help="Check COCO records and photo availability")
    p.add_argument("--smoke", action="store_true", help="Verify model shapes with random tensors; no photos or downloads")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    if args.smoke:
        smoke_test(device)
        return
    data = LaptopCOCO(args.images, args.annotations)
    if args.report_dataset:
        missing = data.check_images()
        print(f"COCO contains {len(data)} image entries, {len(data.coco['annotations'])} annotations, "
              f"categories: {data.coco.get('categories')}")
        print(f"Source photos: {len(data)-len(missing)} present; {len(missing)} missing")
        if missing:
            print("Missing examples:")
            for _, p in missing[:5]:
                print("  ", p)
        return
    missing = data.check_images()
    if missing:
        raise FileNotFoundError(f"Found {len(missing)} missing source photographs. "
                                f"Place them in {args.images}/set1, set2 and set3.\n" +
                                "\n".join(x[1] for x in missing[:8]))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    split = load_or_create_split(data, args.results_dir / "split_ids.json", seed=args.seed)
    print("Split sizes:", {k: len(split[k]) for k in ("train", "val", "test")})
    if args.prepare:
        yaml_path = export_ultralytics_yolo(data, split, args.results_dir / "prepared_yolo")
        print("Ultralytics dataset prepared:", yaml_path)
        return
    experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    for experiment in experiments:
        out_dir = args.results_dir / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 70 + "\nTraining:", experiment, "device:", device)
        if experiment == "pretrained-yolo":
            metrics, predictions, test_data = train_pretrained_yolo(args, data, split, device, out_dir)
        else:
            metrics, predictions, test_data = train_torch_model(experiment, args, data, split, device, out_dir)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        representative_id = split["test"][0]
        save_prediction_pair(test_data, representative_id, predictions[representative_id], out_dir / "prediction.png")
        print(experiment, "mAP@0.50:", f"{metrics['mAP50']:.3f}", "precision:",
              f"{metrics['precision']:.3f}", "recall:", f"{metrics['recall']:.3f}")
    print("Completed requested experiment(s). To compare all four, run: python compare_results.py")


if __name__ == "__main__":
    main()
