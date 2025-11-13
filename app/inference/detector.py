"""
Detector de balón optimizado usando RF-DETR.

Optimizaciones para baja latencia:
- RF-DETR Medium con optimize_for_inference
- Soporte CUDA automático
- Half-precision (FP16) en GPU
- Batch size = 1 para streaming
- Configuración de confidence threshold adaptativa
"""

import numpy as np
import torch
import logging
import time
import supervision as sv
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
from collections import deque
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class BallDetector:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        half_precision: bool = True,
        imgsz: int = 640,
        enable_tensorrt: bool = False,
        multi_scale: bool = False,
        warmup_iterations: int = 3
    ):
        try:
            from rfdetr import RFDETRMedium
        except ImportError:
            raise ImportError("RF-DETR not installed. Run: pip install rfdetr supervision")
        
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.enable_tensorrt = enable_tensorrt
        self.multi_scale = multi_scale
        self.warmup_iterations = warmup_iterations
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.half_precision = half_precision and self.device == 'cuda'
        
        logger.info(f"Initializing BallDetector with RF-DETR Medium")
        if self.model_path:
            logger.info(f"Custom model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Half precision: {self.half_precision}")
        logger.info(f"Image size: {self.imgsz}")
        
        from rfdetr import RFDETRMedium
        
        logger.info("Initializing RF-DETR Medium model...")
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading fine-tuned checkpoint: {self.model_path}")
            # RF-DETR handles device internally - don't pass device parameter
            self.model = RFDETRMedium(
                num_classes=1,
                resolution=self.imgsz,
                pretrain_weights=str(self.model_path)
            )
            logger.info("Custom model weights loaded successfully")
        else:
            logger.warning("No custom model found, using COCO pretrained weights")
            self.model = RFDETRMedium(resolution=self.imgsz)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("CUDA optimizations enabled")
        
        logger.info("Applying optimize_for_inference() - CRITICAL STEP")
        self.model.optimize_for_inference()
        logger.info("Model optimization complete")
        
        # Apply torch.compile if PyTorch 2.0+ (additional 30% speedup)
        if hasattr(torch, '__version__') and torch.__version__ >= "2.0":
            if hasattr(self.model, 'model'):
                try:
                    logger.info("Applying torch.compile() for additional speedup...")
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",  # Optimized for repeated small inputs
                        fullgraph=False  # Allow fallback for dynamic parts
                    )
                    logger.info("✓ torch.compile() applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply torch.compile(): {e}")
        
        logger.info(f"✓ RF-DETR ready on device: {self.device}")
        logger.info("Note: RF-DETR handles GPU internally, half_precision managed by optimize_for_inference()")
        
        self.inference_times = deque(maxlen=100)
        self.detection_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=100)
        
        self.stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'empty_frames': 0,
            'avg_confidence': 0.0
        }
        
        self._warmup()
        
        logger.info("BallDetector initialized successfully")
    
    def _warmup(self):
        logger.info(f"Warming up model ({self.warmup_iterations} iterations)...")
        dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_frame)
        
        warmup_times = []
        with torch.no_grad():
            for i in range(self.warmup_iterations):
                start = time.time()
                _ = self.model.predict(dummy_image, threshold=self.confidence_threshold)
                elapsed = (time.time() - start) * 1000
                warmup_times.append(elapsed)
                logger.debug(f"Warmup {i+1}/{self.warmup_iterations}: {elapsed:.2f}ms")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        avg_warmup = np.mean(warmup_times)
        logger.info(f"Warmup completed - avg time: {avg_warmup:.2f}ms")
    
    def predict(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        return_raw: bool = False,
        augment: bool = False
    ) -> List[Tuple[float, float, float, float, float, int]]:
        start_time = time.time()
        self.stats['total_inferences'] += 1
        
        # GPU path: accept torch.Tensor directly (zero-copy, already in VRAM)
        if isinstance(frame, torch.Tensor):
            # Tensor should be [3, H, W] or [H, W, 3] in RGB, float [0..1]
            if frame.dim() == 3:
                if frame.shape[0] == 3:  # [3, H, W] → [H, W, 3]
                    frame = frame.permute(1, 2, 0)
                
                # Ensure float [0..1]
                if frame.dtype == torch.uint8:
                    frame = frame.float() / 255.0
                
                # Convert to numpy for RF-DETR (TODO: RF-DETR native tensor support)
                # This is a CPU copy, but only for the resized image (not original frame)
                frame_rgb = (frame * 255).byte().cpu().numpy()
                image = frame_rgb
            else:
                raise ValueError(f"Invalid tensor shape: {frame.shape}. Expected [3, H, W] or [H, W, 3]")
        
        # CPU path: numpy array
        elif isinstance(frame, np.ndarray):
            # RF-DETR expects RGB, so flip BGR if needed
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3:
                # BGR to RGB using cv2 (faster than [::-1].copy())
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = frame_rgb
        else:
            # Already PIL or other format
            image = frame
        
        with torch.no_grad():
            if self.multi_scale:
                detections_sv = self._multi_scale_inference(image)
            else:
                detections_sv = self.model.predict(image, threshold=self.confidence_threshold)
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        if self.stats['total_inferences'] % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.stats['total_inferences'] % 20 == 0:
            n = 0 if detections_sv is None else len(detections_sv)
            top_conf = 0.0 if n == 0 else float(np.max(detections_sv.confidence))
            logger.debug(f"Detections: {n} | top_conf: {top_conf:.3f} | thr: {self.confidence_threshold:.3f}")

        if return_raw:
            return detections_sv
        
        detections = self._parse_sv_detections(detections_sv)
        
        if len(detections) == 0:
            self.stats['empty_frames'] += 1
        else:
            self.stats['total_detections'] += len(detections)
            for det in detections:
                self.confidence_history.append(det[4])
        
        if self.stats['total_inferences'] % 100 == 0:
            avg_time = np.mean(self.inference_times)
            avg_conf = np.mean(self.confidence_history) if self.confidence_history else 0
            logger.debug(f"Detector stats: inferences={self.stats['total_inferences']}, "
                        f"avg_time={avg_time:.2f}ms, avg_conf={avg_conf:.3f}")
        
        return detections
    
    def _parse_sv_detections(self, detections_sv: sv.Detections) -> List[Tuple[float, float, float, float, float, int]]:
        detections = []
        
        if detections_sv is None or len(detections_sv) == 0:
            return detections
        
        for i in range(len(detections_sv)):
            x1, y1, x2, y2 = detections_sv.xyxy[i]
            conf = float(detections_sv.confidence[i])
            cls_id = int(detections_sv.class_id[i])
            
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            calibrated_conf = self._calibrate_confidence(conf, width, height)
            
            detections.append((
                x_center,
                y_center,
                width,
                height,
                calibrated_conf,
                cls_id
            ))
        
        return detections
    
    def _multi_scale_inference(self, image: Image.Image) -> sv.Detections:
        scales = [0.8, 1.0, 1.2]
        all_detections = []
        
        original_size = image.size
        
        for scale in scales:
            new_w = int(original_size[0] * scale)
            new_h = int(original_size[1] * scale)
            resized_image = image.resize((new_w, new_h), Image.BILINEAR)
            
            detections = self.model.predict(resized_image, threshold=self.confidence_threshold)
            
            if detections is not None and len(detections) > 0:
                detections.xyxy = detections.xyxy / scale
                all_detections.append(detections)
        
        if len(all_detections) == 0:
            return sv.Detections.empty()
        
        return self._merge_multi_scale_detections(all_detections)
    
    def _merge_multi_scale_detections(self, detections_list: List[sv.Detections]) -> sv.Detections:
        if len(detections_list) == 1:
            return detections_list[0]
        
        all_xyxy = np.vstack([d.xyxy for d in detections_list])
        all_conf = np.concatenate([d.confidence for d in detections_list])
        all_cls = np.concatenate([d.class_id for d in detections_list])
        
        merged = sv.Detections(
            xyxy=all_xyxy,
            confidence=all_conf,
            class_id=all_cls
        )
        
        nms_idx = self._apply_nms(merged)
        
        return sv.Detections(
            xyxy=merged.xyxy[nms_idx],
            confidence=merged.confidence[nms_idx],
            class_id=merged.class_id[nms_idx]
        )
    
    def _apply_nms(self, detections: sv.Detections) -> np.ndarray:
        boxes = torch.from_numpy(detections.xyxy).float()
        scores = torch.from_numpy(detections.confidence).float()
        
        from torchvision.ops import nms
        keep_idx = nms(boxes, scores, self.iou_threshold)
        
        return keep_idx.cpu().numpy()
    
    def _calibrate_confidence(self, conf: float, width: float, height: float) -> float:
        area = width * height
        
        if area < 3000:
            calibrated = conf * 1.10
        elif area > 150000:
            calibrated = conf * 0.95
        else:
            calibrated = conf
        
        return np.clip(calibrated, 0.0, 1.0)
    
    def predict_ball_only(
        self,
        frame: Union[np.ndarray, torch.Tensor],
        ball_class_id: int = 0,
        use_temporal_filtering: bool = True,
        return_candidates: bool = False
    ) -> Optional[Tuple[float, float, float, float, float]]:
        detections = self.predict(frame)
        
        unique_classes = set(d[5] for d in detections) if detections else set()
        if len(unique_classes) <= 1:
            ball_detections = detections
        else:
            ball_detections = [d for d in detections if d[5] == ball_class_id]
        
        if len(ball_detections) == 0:
            if use_temporal_filtering and len(self.detection_history) > 0:
                logger.debug("No detection, using temporal prediction")
                predicted = self._predict_from_history()
                if return_candidates:
                    return predicted, detections
                return predicted
            if return_candidates:
                return None, detections
            return None
        
        if len(ball_detections) == 1:
            best_detection = ball_detections[0]
        else:
            best_detection = self._select_best_detection(ball_detections)
        
        detection_tuple = best_detection[:5]
        
        if use_temporal_filtering:
            self.detection_history.append(detection_tuple)
        
        if return_candidates:
            return detection_tuple, detections
        return detection_tuple
    
    def _select_best_detection(self, detections: List[Tuple]) -> Tuple:
        if len(self.detection_history) == 0:
            return max(detections, key=lambda x: x[4])
        
        last_pos = np.array([self.detection_history[-1][0], self.detection_history[-1][1]])
        
        scores = []
        for det in detections:
            conf_score = det[4]
            
            pos = np.array([det[0], det[1]])
            distance = np.linalg.norm(pos - last_pos)
            proximity_score = 1.0 / (1.0 + distance / 100.0)
            
            combined_score = 0.6 * conf_score + 0.4 * proximity_score
            scores.append(combined_score)
        
        best_idx = np.argmax(scores)
        return detections[best_idx]
    
    def _predict_from_history(self) -> Optional[Tuple[float, float, float, float, float]]:
        if len(self.detection_history) < 2:
            return None
        
        recent = list(self.detection_history)[-3:]
        
        x_vals = [d[0] for d in recent]
        y_vals = [d[1] for d in recent]
        
        pred_x = x_vals[-1] + (x_vals[-1] - x_vals[-2]) if len(recent) >= 2 else x_vals[-1]
        pred_y = y_vals[-1] + (y_vals[-1] - y_vals[-2]) if len(recent) >= 2 else y_vals[-1]
        
        avg_w = np.mean([d[2] for d in recent])
        avg_h = np.mean([d[3] for d in recent])
        avg_conf = np.mean([d[4] for d in recent]) * 0.5
        
        return (pred_x, pred_y, avg_w, avg_h, avg_conf)
    
    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {threshold:.3f}")
    
    def auto_tune_confidence(self):
        if len(self.confidence_history) < 50:
            return
        
        median_conf = np.median(self.confidence_history)
        std_conf = np.std(self.confidence_history)
        
        if median_conf > 0.8 and std_conf < 0.1:
            new_threshold = min(self.confidence_threshold + 0.05, 0.5)
            logger.info(f"Auto-tuning: increasing threshold to {new_threshold:.3f}")
            self.set_confidence_threshold(new_threshold)
        elif median_conf < 0.4:
            new_threshold = max(self.confidence_threshold - 0.05, 0.15)
            logger.info(f"Auto-tuning: decreasing threshold to {new_threshold:.3f}")
            self.set_confidence_threshold(new_threshold)
    
    def get_model_info(self) -> Dict:
        info = {
            'model_type': 'RF-DETR Medium',
            'model_path': str(self.model_path) if self.model_path else 'Pretrained COCO',
            'device': self.device,
            'half_precision': self.half_precision,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'image_size': self.imgsz,
            'multi_scale': self.multi_scale,
            'optimized_for_inference': True,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return info
    
    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        
        if self.inference_times:
            stats['avg_inference_time_ms'] = np.mean(self.inference_times)
            stats['min_inference_time_ms'] = np.min(self.inference_times)
            stats['max_inference_time_ms'] = np.max(self.inference_times)
        
        if self.confidence_history:
            stats['avg_confidence'] = np.mean(self.confidence_history)
            stats['median_confidence'] = np.median(self.confidence_history)
        
        return stats
    
    def reset_stats(self):
        self.stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'empty_frames': 0,
            'avg_confidence': 0.0
        }
        self.inference_times.clear()
        self.confidence_history.clear()
        logger.info("Detector stats reset")
