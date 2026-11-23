from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor


class LaptopCOCO(Dataset):

    def __init__(self, images_root: str | Path, annotation_file: str | Path, image_ids=None):
        self.images_root = Path(images_root)
        self.annotation_file = Path(annotation_file)
        self.coco = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.by_id = {int(i["id"]): i for i in self.coco["images"]}
        self.ann_by_image = defaultdict(list)
        for ann in self.coco["annotations"]:
            if int(ann.get("iscrowd", 0)) == 0 and int(ann["category_id"]) == 1:
                self.ann_by_image[int(ann["image_id"])].append(ann)
        self.image_ids = sorted(self.by_id) if image_ids is None else list(map(int, image_ids))
        if not self.image_ids:
            raise ValueError("No image entries selected")

    def __len__(self):
        return len(self.image_ids)

    def path_for(self, image_id):
        
        name = self.by_id[int(image_id)]["file_name"].replace("\\", "/")
        path = (self.images_root / name).resolve()
        if not path.is_relative_to(self.images_root.resolve()):
            raise ValueError(f"Unsafe file_name in COCO file: {name}")
        return path

    def check_images(self):
        return [(i, str(self.path_for(i))) for i in self.image_ids if not self.path_for(i).is_file()]

    def target_for(self, image_id, size):
        width, height = size
        boxes = []
        for ann in self.ann_by_image[int(image_id)]:
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            x1, y1 = max(0, min(width, x)), max(0, min(height, y))
            x2, y2 = max(0, min(width, x + w)), max(0, min(height, y + h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return {
            "boxes": b,
            "labels": torch.ones(len(b), dtype=torch.int64),
            "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
            "area": (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]),
            "iscrowd": torch.zeros(len(b), dtype=torch.int64),
        }

    def __getitem__(self, ix):
        image_id = self.image_ids[ix]
        path = self.path_for(image_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Copy the source image folders into raw_images/")
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            target = self.target_for(image_id, rgb.size)
            tensor = to_tensor(rgb)
        return tensor, target


def stable_split(dataset: LaptopCOCO, seed=42, test_fraction=0.20, val_fraction_of_train=0.10):
    ids = sorted(dataset.by_id)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = round(len(ids) * test_fraction)
    test = sorted(ids[:n_test])
    pool = ids[n_test:]
    n_val = round(len(pool) * val_fraction_of_train)
    val = sorted(pool[:n_val])
    train = sorted(pool[n_val:])
    return {"train": train, "val": val, "test": test, "training_pool": sorted(pool), "seed": seed}


def save_split(split, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_or_create_split(dataset, path, seed=42):
    path = Path(path)
    split = stable_split(dataset, seed=seed)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in ("train", "val", "test")} != {k: split[k] for k in ("train", "val", "test")}:
            raise ValueError(f"Existing split at {path} differs. Remove it only if intentionally repartitioning.")
        return old
    save_split(split, path)
    return split


def collate_detection(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def to_device(target, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}


def export_ultralytics_yolo(dataset: LaptopCOCO, split: dict, out_dir):
    out_dir = Path(out_dir).resolve()
    missing = dataset.check_images()
    if missing:
        preview = "\n".join(f"  {p}" for _, p in missing[:10])
        raise FileNotFoundError(f"{len(missing)} images missing; cannot export YOLO data. Examples:\n{preview}")
    manifest = {}
    for part in ("train", "val", "test"):
        image_dir = out_dir / "images" / part
        label_dir = out_dir / "labels" / part
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id in split[part]:
            src = dataset.path_for(image_id)
            stem = f"image_{image_id:05d}"
            dest = image_dir / (stem + src.suffix.lower())
            if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dest)
            with Image.open(src) as pil:
                w, h = pil.size
            target = dataset.target_for(image_id, (w, h))
            lines = []
            for x1, y1, x2, y2 in target["boxes"].tolist():
                xc = (x1 + x2) / (2 * w)
                yc = (y1 + y2) / (2 * h)
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            manifest[stem] = str(src)
    
    root = out_dir.as_posix()
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(f'path: "{root}"\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: laptop\n', encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data_yaml


def evaluate_ap50(predictions: dict, targets: dict, score_threshold=0.05, iou_threshold=0.5):
    ranked = []
    n_gt = 0
    for image_id, gt in targets.items():
        gt = torch.as_tensor(gt).float().reshape(-1, 4)
        n_gt += len(gt)
        pred = predictions.get(image_id, {})
        boxes = torch.as_tensor(pred.get("boxes", [])).float().reshape(-1, 4)
        scores = torch.as_tensor(pred.get("scores", [])).float().reshape(-1)
        for score, box in zip(scores.tolist(), boxes.tolist()):
            if score >= score_threshold:
                ranked.append((float(score), image_id, box))
    ranked.sort(key=lambda rec: -rec[0])
    matched = {key: set() for key in targets}
    tp, fp = [], []
    for score, image_id, box in ranked:
        gt = torch.as_tensor(targets[image_id]).float().reshape(-1, 4)
        if len(gt) == 0:
            tp.append(0); fp.append(1); continue
        ious = box_iou(torch.as_tensor(box).float().reshape(1, 4), gt)[0]
        for used in matched[image_id]:
            ious[used] = -1
        index = int(ious.argmax())
        if float(ious[index]) >= iou_threshold:
            matched[image_id].add(index)
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    cumul_tp = np.cumsum(tp)
    cumul_fp = np.cumsum(fp)
    precision_curve = cumul_tp / np.maximum(cumul_tp + cumul_fp, 1)
    recall_curve = cumul_tp / max(n_gt, 1)
    
    if n_gt == 0:
        ap = 0.0
    else:
        recall = np.concatenate(([0.0], recall_curve, [1.0]))
        prec = np.concatenate(([0.0], precision_curve, [0.0]))
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        change = np.where(recall[1:] != recall[:-1])[0]
        ap = float(np.sum((recall[change + 1] - recall[change]) * prec[change + 1]))
    tps, fps = int(sum(tp)), int(sum(fp))
    return {
        "mAP50": ap, "AP50_laptop": ap,
        "precision": tps / max(tps + fps, 1),
        "recall": tps / max(n_gt, 1),
        "true_positives": tps, "false_positives": fps,
        "false_negatives": n_gt - tps, "gt_instances": n_gt,
        "evaluated_images": len(targets), "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "metric_definition": "single-class VOC-style area under interpolated PR curve at IoU=0.50; NOT COCO mAP@[.50:.95]",
    }


def save_prediction_pair(dataset, image_id, pred, output_path, score_threshold=0.25):
    path = dataset.path_for(image_id)
    with Image.open(path) as im:
        img = im.convert("RGB")
    ground = img.copy(); estimate = img.copy()
    left = ImageDraw.Draw(ground); right = ImageDraw.Draw(estimate)
    for bbox in dataset.target_for(image_id, img.size)["boxes"].tolist():
        left.rectangle(bbox, outline="#00c853", width=3)
    p_boxes = torch.as_tensor(pred.get("boxes", [])).reshape(-1, 4)
    p_scores = torch.as_tensor(pred.get("scores", [])).reshape(-1)
    for bbox, score in zip(p_boxes.tolist(), p_scores.tolist()):
        if score >= score_threshold:
            right.rectangle(bbox, outline="#ff6d00", width=3)
            right.text((bbox[0], max(0, bbox[1] - 12)), f"laptop {score:.2f}", fill="#ff6d00")
    canvas = Image.new("RGB", (img.width * 2, img.height + 25), "white")
    canvas.paste(ground, (0, 25)); canvas.paste(estimate, (img.width, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Ground truth", fill="black")
    draw.text((img.width + 8, 7), "Predictions", fill="black")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
