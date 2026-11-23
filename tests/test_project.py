from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image

from dataset_utils import LaptopCOCO, stable_split, export_ultralytics_yolo, evaluate_ap50
from models import build_custom_ssd, TinyYOLO, make_yolo_targets, yolo_loss, yolo_resize_batch

ROOT = Path(__file__).resolve().parents[1]


def test_original_coco_structure_and_split():
    data = LaptopCOCO(ROOT / "raw_images", ROOT / "laptop_all_3_batches_COCO.json")
    assert len(data) == 94
    assert len(data.coco["annotations"]) == 100
    split = stable_split(data)
    assert tuple(len(split[k]) for k in ("train", "val", "test")) == (67, 8, 19)
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["test"]) & set(split["train"]))


def test_metric_single_correct_prediction():
    preds = {1: {"boxes": [[20, 20, 70, 70], [100, 100, 110, 110]], "scores": [0.9, 0.2]}}
    gt = {1: [[20, 20, 70, 70]]}
    metrics = evaluate_ap50(preds, gt)
    assert abs(metrics["mAP50"] - 1.0) < 1e-6
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_dataset_can_export_yolo(tmp_path):
    src = tmp_path / "raw_images"
    (src / "set1").mkdir(parents=True)
    images, annotations = [], []
    for i in range(1, 6):
        Image.fromarray(np.full((128, 192, 3), 50 + i * 20, np.uint8)).save(src / "set1" / f"image{i}.jpg")
        images.append({"id": i, "width": 192, "height": 128, "file_name": f"set1/image{i}.jpg"})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [40, 20, 80, 90], "iscrowd": 0})
    ann = tmp_path / "data.json"
    ann.write_text(json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "laptop"}]}))
    data = LaptopCOCO(src, ann)
    split = stable_split(data)
    yaml_path = export_ultralytics_yolo(data, split, tmp_path / "yolo")
    assert yaml_path.exists()
    all_labels = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(all_labels) == 5
    for p in all_labels:
        fields = p.read_text().strip().split()
        assert fields[0] == "0"
        assert len(fields) == 5
        assert all(0 <= float(v) <= 1 for v in fields[1:])


def test_yolo_targets_trainable():
    model = TinyYOLO(416)
    image = torch.randn(1, 3, 416, 416)
    raw = model(image)
    assert raw.shape == (1, 13, 13, 3, 6)
    target = make_yolo_targets([{"boxes": torch.tensor([[20., 40., 170., 260.]])}], 416,
                               model.anchors, 13, torch.device("cpu"))
    assert int(target[..., 4].sum().item()) == 1
    loss, _ = yolo_loss(raw, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head[-1].weight.grad is not None
    with torch.no_grad():
        predictions = model.predict([torch.rand(3, 256, 320)], score_threshold=0.9)
    assert predictions[0]["boxes"].shape[-1] == 4


def test_yolo_resize_boxes():
    image = torch.zeros(3, 200, 400)
    targets = [{"boxes": torch.tensor([[100, 50, 200, 150]], dtype=torch.float32)}]
    _, t = yolo_resize_batch([image], targets, 416, torch.device("cpu"))
    assert torch.allclose(t[0]["boxes"], torch.tensor([[104., 104., 208., 312.]]))


def test_ssd_training_loss_backward():
    
    model = build_custom_ssd()
    model.train()
    image = torch.rand(3, 300, 300)
    targets = [{"boxes": torch.tensor([[30., 35., 180., 230.]]),
                "labels": torch.tensor([1], dtype=torch.long)}]
    losses = model([image], targets)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.head.class_heads[0].weight.grad is not None
