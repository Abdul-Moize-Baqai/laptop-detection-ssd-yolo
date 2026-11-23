from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection.ssd import SSD, SSDClassificationHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
from torchvision.ops import nms


class TinySSDBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.b1 = block(3, 16)
        self.b2 = block(16, 32)
        self.b3 = block(32, 64)
        self.b4 = block(64, 128)
        self.b5 = block(128, 128)
        self.out_channels = [64, 128, 128]

    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        high = x
        mid = self.b4(high)
        low = self.b5(mid)
        return OrderedDict([("0", high), ("1", mid), ("2", low)])


class CustomSSDHeads(nn.Module):
    def __init__(self, feature_channels, anchors_per_location, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.class_heads = nn.ModuleList([nn.Conv2d(c, a * num_classes, 3, padding=1)
                                          for c, a in zip(feature_channels, anchors_per_location)])
        self.box_heads = nn.ModuleList([nn.Conv2d(c, a * 4, 3, padding=1)
                                        for c, a in zip(feature_channels, anchors_per_location)])

    def forward(self, features):
        classes, boxes = [], []
        for feat, class_layer, box_layer in zip(features, self.class_heads, self.box_heads):
            batch = feat.shape[0]
            cls = class_layer(feat).permute(0, 2, 3, 1).contiguous().view(batch, -1, self.num_classes)
            loc = box_layer(feat).permute(0, 2, 3, 1).contiguous().view(batch, -1, 4)
            classes.append(cls)
            boxes.append(loc)
        return {"cls_logits": torch.cat(classes, dim=1),
                "bbox_regression": torch.cat(boxes, dim=1)}


def build_custom_ssd():
    backbone = TinySSDBackbone()
    anchors = DefaultBoxGenerator(
        aspect_ratios=[[2], [2, 3], [2, 3]],
        scales=[0.05, 0.15, 0.35, 0.60],
        clip=True,
    )
    prediction_head = CustomSSDHeads(backbone.out_channels, anchors.num_anchors_per_location(), num_classes=2)
    return SSD(
        backbone=backbone,
        anchor_generator=anchors,
        head=prediction_head,
        size=(300, 300),
        num_classes=2,  
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=100,
    )


def build_pretrained_ssd(download_weights=True):
    if download_weights:
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    else:
        model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=91)
    old = model.head.classification_head
    channels = [conv.in_channels for conv in old.module_list]
    anchors = model.anchor_generator.num_anchors_per_location()
    new_head = SSDClassificationHead(channels, anchors, num_classes=2)
    if download_weights:
        
        
        laptop_id = SSD300_VGG16_Weights.DEFAULT.meta["categories"].index("laptop")
        old_classes = len(SSD300_VGG16_Weights.DEFAULT.meta["categories"])
        with torch.no_grad():
            for old_conv, new_conv, n_anchor in zip(old.module_list, new_head.module_list, anchors):
                new_conv.weight.copy_(old_conv.weight.view(n_anchor, old_classes, *old_conv.weight.shape[1:])
                                      [:, [0, laptop_id]].reshape_as(new_conv.weight))
                new_conv.bias.copy_(old_conv.bias.view(n_anchor, old_classes)
                                    [:, [0, laptop_id]].reshape_as(new_conv.bias))
    model.head.classification_head = new_head
    return model


def conv_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1),
                         nn.BatchNorm2d(co), nn.LeakyReLU(0.1, inplace=True),
                         nn.MaxPool2d(2))


class TinyYOLO(nn.Module):
    def __init__(self, input_size=416, anchors=None):
        super().__init__()
        if input_size % 32:
            raise ValueError("YOLO image size must be a multiple of 32")
        self.input_size = int(input_size)
        if anchors is None:
            anchors = [(28, 35), (85, 100), (180, 190)]
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.backbone = nn.Sequential(
            conv_block(3, 16), conv_block(16, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, len(anchors) * 6, kernel_size=1),
        )

    def forward(self, x):
        x = self.head(self.backbone(x))
        b, _, s, _ = x.shape
        return x.permute(0, 2, 3, 1).contiguous().view(b, s, s, len(self.anchors), 6)

    @torch.no_grad()
    def predict(self, images, score_threshold=0.05, nms_iou=0.45):
        device = next(self.parameters()).device
        source_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        batch = torch.stack([F.interpolate(im[None], size=(self.input_size, self.input_size),
                                            mode="bilinear", align_corners=False)[0] for im in images]).to(device)
        prediction = self(batch)
        b, s, _, a, _ = prediction.shape
        ys, xs = torch.meshgrid(torch.arange(s, device=device), torch.arange(s, device=device), indexing="ij")
        xs = xs.view(1, s, s, 1)
        ys = ys.view(1, s, s, 1)
        cell = self.input_size / s
        cx = (torch.sigmoid(prediction[..., 0]) + xs) * cell
        cy = (torch.sigmoid(prediction[..., 1]) + ys) * cell
        wh = torch.exp(prediction[..., 2:4].clamp(-6, 6)) * self.anchors.view(1, 1, 1, a, 2)
        scores = torch.sigmoid(prediction[..., 4]) * torch.sigmoid(prediction[..., 5])
        boxes = torch.stack([cx - wh[..., 0] / 2, cy - wh[..., 1] / 2,
                             cx + wh[..., 0] / 2, cy + wh[..., 1] / 2], dim=-1)
        outputs = []
        for i, ((orig_h, orig_w), bxs, srs) in enumerate(zip(source_sizes, boxes, scores)):
            bxs = bxs.reshape(-1, 4)
            srs = srs.reshape(-1)
            valid = srs >= score_threshold
            bxs, srs = bxs[valid], srs[valid]
            if len(srs):
                bxs[:, [0, 2]] = bxs[:, [0, 2]].clamp(0, self.input_size)
                bxs[:, [1, 3]] = bxs[:, [1, 3]].clamp(0, self.input_size)
                valid_box = (bxs[:, 2] > bxs[:, 0]) & (bxs[:, 3] > bxs[:, 1])
                bxs, srs = bxs[valid_box], srs[valid_box]
            kept = nms(bxs, srs, nms_iou)[:100]
            bxs, srs = bxs[kept], srs[kept]
            if len(bxs):
                bxs[:, [0, 2]] *= orig_w / self.input_size
                bxs[:, [1, 3]] *= orig_h / self.input_size
            outputs.append({"boxes": bxs.cpu(), "scores": srs.cpu(),
                            "labels": torch.ones(len(srs), dtype=torch.int64)})
        return outputs


def anchor_wh_iou(gt_wh, anchor_wh):
    intersections = torch.minimum(gt_wh, anchor_wh).prod(-1)
    unions = gt_wh.prod() + anchor_wh.prod(-1) - intersections
    return intersections / unions.clamp_min(1e-8)


def make_yolo_targets(targets, size, anchor_wh, grid_size, device):
    a = len(anchor_wh)
    result = torch.zeros((len(targets), grid_size, grid_size, a, 6), device=device)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(device)
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            center_x, center_y = ((x1 + x2) / 2) / size * grid_size, ((y1 + y2) / 2) / size * grid_size
            gx, gy = int(center_x.clamp(0, grid_size - 1e-4)), int(center_y.clamp(0, grid_size - 1e-4))
            preference = torch.argsort(anchor_wh_iou(torch.stack([w, h]), anchor_wh), descending=True)
            ai = None
            for candidate in preference.tolist():
                if result[batch_index, gy, gx, candidate, 4] == 0:
                    ai = candidate
                    break
            if ai is None:
                continue  
            result[batch_index, gy, gx, ai, 0] = center_x - gx
            result[batch_index, gy, gx, ai, 1] = center_y - gy
            result[batch_index, gy, gx, ai, 2:4] = torch.log(torch.stack([w, h]) / anchor_wh[ai])
            result[batch_index, gy, gx, ai, 4] = 1
            result[batch_index, gy, gx, ai, 5] = 1
    return result


def yolo_loss(logits, targets):
    positive = targets[..., 4] > 0.5
    negative = ~positive
    obj_logits = logits[..., 4]
    obj_tgt = targets[..., 4]
    pos_obj = F.binary_cross_entropy_with_logits(obj_logits[positive], obj_tgt[positive]) if positive.any() else logits.sum() * 0
    neg_obj = F.binary_cross_entropy_with_logits(obj_logits[negative], obj_tgt[negative]) if negative.any() else logits.sum() * 0
    if positive.any():
        coord_xy = F.binary_cross_entropy_with_logits(logits[..., 0:2][positive], targets[..., 0:2][positive])
        coord_wh = F.smooth_l1_loss(logits[..., 2:4][positive], targets[..., 2:4][positive])
        class_loss = F.binary_cross_entropy_with_logits(logits[..., 5][positive], targets[..., 5][positive])
    else:
        coord_xy = coord_wh = class_loss = logits.sum() * 0
    return (5 * coord_xy + 3 * coord_wh + class_loss + pos_obj + 0.5 * neg_obj), {
        "xy": float(coord_xy.detach()), "wh": float(coord_wh.detach()),
        "class": float(class_loss.detach()), "object": float((pos_obj + 0.5 * neg_obj).detach()),
    }


def yolo_resize_batch(images, targets, size, device):
    resized, converted = [], []
    for img, tgt in zip(images, targets):
        _, h, w = img.shape
        resized.append(F.interpolate(img[None], size=(size, size), mode="bilinear", align_corners=False)[0])
        scaled = tgt["boxes"].clone().to(device)
        if len(scaled):
            scaled[:, [0, 2]] *= size / w
            scaled[:, [1, 3]] *= size / h
        converted.append({"boxes": scaled})
    return torch.stack(resized).to(device), converted
