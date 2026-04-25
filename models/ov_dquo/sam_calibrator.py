import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


class SamCalibrator:
    def __init__(self, conf=0.3, iou_threshold=0.3, boost_scale=0.15, penalty_scale=0.05):
        self.conf = conf
        self.iou_threshold = iou_threshold
        self.boost_scale = boost_scale
        self.penalty_scale = penalty_scale
        self._model = None
        # Single-entry cache: (path_str, sam_boxes, masks_28, sam_confs) or (path_str, None, None, None)
        self._fastsam_cache = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import FastSAM
            self._model = FastSAM("FastSAM-x.pt")
        return self._model

    def _run_fastsam(self, image_path):
        """Run FastSAM with mask output and cache the result.

        Returns (sam_boxes_xyxy, masks_28x28, sam_confs) or None on failure.
        Masks are (N, 28, 28) bool tensors: full-image binary masks downsampled to 28×28,
        matching the coordinate system of precomputed sam_priors .pt files.
        """
        path_str = str(image_path)
        if self._fastsam_cache is not None and self._fastsam_cache[0] == path_str:
            cached = self._fastsam_cache[1:]
            return None if cached[0] is None else cached
        try:
            model = self._load_model()
            results = model(
                path_str, conf=self.conf, iou=0.6, imgsz=1024,
                retina_masks=True, verbose=False,
            )
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                self._fastsam_cache = (path_str, None, None, None)
                return None
            sam_boxes = result.boxes.xyxy.cpu().float()   # (N, 4)
            sam_confs = result.boxes.conf.cpu().float()   # (N,)
            if result.masks is not None:
                masks_hw = result.masks.data.cpu().float()   # (N, H, W)
                masks_28 = (
                    F.interpolate(masks_hw.unsqueeze(1), size=(28, 28), mode='nearest')
                    .squeeze(1)
                    .bool()
                )  # (N, 28, 28)
            else:
                masks_28 = torch.ones(len(sam_boxes), 28, 28, dtype=torch.bool)
            self._fastsam_cache = (path_str, sam_boxes, masks_28, sam_confs)
            return sam_boxes, masks_28, sam_confs
        except Exception:
            self._fastsam_cache = (path_str, None, None, None)
            return None

    def get_masks(self, image_path, pred_boxes_xyxy):
        """Run FastSAM and return (sam_conf_per_pred, matched_masks_28x28) or None on failure.

        sam_conf_per_pred: (N_pred,) float, best SAM confidence for each pred_box
                           (0 for pred_boxes with no SAM match above iou_threshold).
        matched_masks_28x28: (N_pred, 28, 28) bool, full-image SAM mask of the best-IoU
                              matching SAM detection per pred_box; ones (neutral) for unmatched.
        The 28×28 coordinate system is identical to the precomputed sam_priors masks so that
        crop_and_resize_masks can use them directly.
        """
        raw = self._run_fastsam(image_path)
        if raw is None:
            return None
        sam_boxes, masks_28, sam_confs = raw
        pred_cpu = pred_boxes_xyxy.cpu().float()
        n_pred = pred_cpu.shape[0]

        ious = box_iou(pred_cpu, sam_boxes)         # (N_pred, N_sam)
        matched = ious > self.iou_threshold          # (N_pred, N_sam)
        has_match = matched.any(dim=1)               # (N_pred,)

        # Best-conf SAM box among those with IoU > threshold (matches original calibrate behavior)
        sam_conf_per_pred = (sam_confs.unsqueeze(0) * matched.float()).max(dim=1).values  # (N_pred,)

        # Per-pred matched masks: best-IoU SAM box per pred, ones for unmatched
        matched_masks = torch.ones(n_pred, 28, 28, dtype=torch.bool)
        if has_match.any():
            best_iou_idx = ious.argmax(dim=1)         # (N_pred,)
            matched_masks[has_match] = masks_28[best_iou_idx[has_match]]

        return sam_conf_per_pred, matched_masks

    def calibrate(self, pred_boxes_xyxy, pred_scores, image_path):
        """
        pred_boxes_xyxy: (N, 4) absolute pixel coords
        pred_scores:     (N,) confidence scores
        Returns calibrated scores, same shape and device as pred_scores.
        Falls back to original scores on any failure.
        """
        if len(pred_scores) == 0:
            return pred_scores
        try:
            result = self.get_masks(image_path, pred_boxes_xyxy.cpu().float())
            if result is None:
                return pred_scores
            sam_conf_per_pred, _ = result
            has_match = sam_conf_per_pred > 0
            calibrated = pred_scores.clone()
            dev = pred_scores.device
            calibrated[has_match] = pred_scores[has_match] * (
                1.0 + self.boost_scale * sam_conf_per_pred[has_match].to(dev)
            )
            calibrated[~has_match] = pred_scores[~has_match] * (1.0 - self.penalty_scale)
            return calibrated
        except Exception:
            return pred_scores
