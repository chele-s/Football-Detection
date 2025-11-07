"""
Detector de balón optimizado usando YOLOv8.

Optimizaciones para baja latencia:
- Soporte CUDA/TensorRT automático
- Half-precision (FP16) en GPU
- Batch size = 1 para streaming
- Configuración de confidence threshold adaptativa
"""

import numpy as np
import torch
import logging
import time
from typing import Optional, List, Tuple, Dict
from pathlib import Path
from collections import deque
import cv2

logger = logging.getLogger(__name__)


class BallDetector:
    
    def __init__(
        self,
        model_path: str,
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
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics")
        
        self.model_path = Path(model_path)
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
        
        logger.info(f"Initializing BallDetector")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Half precision: {self.half_precision}")
        logger.info(f"Image size: {self.imgsz}")
        logger.info(f"TensorRT: {self.enable_tensorrt}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        
        if self.device == 'cuda':
            self.model.to('cuda')
            if self.half_precision:
                self.model.model.half()
                logger.info("FP16 mode enabled")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
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
        
        if self.enable_tensorrt and self.device == 'cuda':
            self._export_tensorrt()
        
        logger.info("BallDetector initialized successfully")
    
    def _warmup(self):
        logger.info(f"Warming up model ({self.warmup_iterations} iterations)...")
        dummy_frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        for i in range(self.warmup_iterations):
            start = time.time()
            _ = self.model.predict(
                dummy_frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.half_precision,
                imgsz=self.imgsz
            )
            elapsed = (time.time() - start) * 1000
            logger.debug(f"Warmup {i+1}/{self.warmup_iterations}: {elapsed:.2f}ms")
        
        logger.info("Warmup completed")
    
    def _export_tensorrt(self):
        try:
            logger.info("Exporting to TensorRT...")
            self.model.export(format='engine', half=self.half_precision, imgsz=self.imgsz)
            logger.info("TensorRT export successful")
        except Exception as e:
            logger.warning(f"TensorRT export failed: {e}. Using standard model.")
    
    def predict(
        self,
        frame: np.ndarray,
        return_raw: bool = False,
        augment: bool = False
    ) -> List[Tuple[float, float, float, float, float, int]]:
        start_time = time.time()
        self.stats['total_inferences'] += 1
        
        if self.multi_scale:
            results = self._multi_scale_inference(frame)
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.half_precision,
                imgsz=self.imgsz,
                augment=augment
            )
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        if return_raw:
            return results
        
        detections = self._parse_results(results)
        
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
    
    def _parse_results(self, results) -> List[Tuple[float, float, float, float, float, int]]:
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                x1, y1, x2, y2 = xyxy
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
    
    def _multi_scale_inference(self, frame: np.ndarray):
        scales = [0.8, 1.0, 1.2]
        all_results = []
        
        for scale in scales:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            
            results = self.model.predict(
                resized,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device,
                half=self.half_precision,
                imgsz=self.imgsz
            )
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                boxes = boxes / scale
                results[0].boxes.xyxy = torch.tensor(boxes).to(results[0].boxes.xyxy.device)
                all_results.append(results[0])
        
        return [self._merge_multi_scale_results(all_results)] if all_results else []
    
    def _merge_multi_scale_results(self, results_list):
        if len(results_list) == 1:
            return results_list[0]
        
        return results_list[0]
    
    def _calibrate_confidence(self, conf: float, width: float, height: float) -> float:
        area = width * height
        
        if area < 100:
            calibrated = conf * 0.8
        elif area > 10000:
            calibrated = conf * 0.9
        else:
            calibrated = conf
        
        return np.clip(calibrated, 0.0, 1.0)
    
    def predict_ball_only(
        self,
        frame: np.ndarray,
        ball_class_id: int = 0,
        use_temporal_filtering: bool = True
    ) -> Optional[Tuple[float, float, float, float, float]]:
        detections = self.predict(frame)
        
        ball_detections = [
            d for d in detections if d[5] == ball_class_id
        ]
        
        if len(ball_detections) == 0:
            if use_temporal_filtering and len(self.detection_history) > 0:
                logger.debug("No detection, using temporal prediction")
                return self._predict_from_history()
            return None
        
        if len(ball_detections) == 1:
            best_detection = ball_detections[0]
        else:
            best_detection = self._select_best_detection(ball_detections)
        
        detection_tuple = best_detection[:5]
        
        if use_temporal_filtering:
            self.detection_history.append(detection_tuple)
        
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
            'model_path': str(self.model_path),
            'device': self.device,
            'half_precision': self.half_precision,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'image_size': self.imgsz,
            'multi_scale': self.multi_scale,
            'tensorrt_enabled': self.enable_tensorrt,
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
