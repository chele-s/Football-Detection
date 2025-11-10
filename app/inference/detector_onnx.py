#!/usr/bin/env python3
"""
ONNX Runtime-accelerated Ball Detector.

Alternative to TensorRT when CUDA versions are incompatible.
Provides 1.5-2x speedup vs PyTorch.
"""

import numpy as np
import cv2
import logging
import time
from typing import Optional, List, Tuple
from pathlib import Path
from collections import deque

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime-gpu: pip install onnxruntime-gpu")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[ONNX] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BallDetectorONNX:
    """ONNX Runtime-powered ball detector"""
    
    def __init__(
        self,
        onnx_path: str,
        confidence_threshold: float = 0.12,
        imgsz: int = 480,
        **kwargs  # Absorb other unused params
    ):
        logger.info(f"🚀 Initializing ONNX Runtime Ball Detector")
        logger.info(f"ONNX model: {onnx_path}")
        
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")
        
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        provider = self.session.get_providers()[0]
        logger.info(f"✓ ONNX Runtime session created")
        logger.info(f"✓ Using provider: {provider}")
        logger.info(f"✓ Input: {self.input_name} shape={self.session.get_inputs()[0].shape}")
        logger.info(f"✓ Outputs: {len(self.output_names)} tensors")
        for i, out in enumerate(self.session.get_outputs()):
            logger.info(f"    Output {i}: {out.name} shape={out.shape}")
        
        # Init stats
        self.inference_times = deque(maxlen=100)
        self.stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'empty_frames': 0
        }
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX model"""
        # Resize
        if frame.shape[0] != self.imgsz or frame.shape[1] != self.imgsz:
            frame = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and CHW format
        frame_chw = (frame_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        
        # Add batch dimension
        return np.expand_dims(frame_chw, axis=0)
    
    def predict(self, frame: np.ndarray, **kwargs) -> List[Tuple]:
        start_time = time.time()
        self.stats['total_inferences'] += 1
        
        input_data = self._preprocess(frame)
        
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )
        
        logger.debug(f"ONNX outputs: {len(outputs)} tensors")
        for i, out in enumerate(outputs):
            logger.debug(f"  Output {i}: shape={out.shape}, dtype={out.dtype}")
        
        detections = self._postprocess(outputs, frame.shape[:2])
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        if detections:
            self.stats['total_detections'] += len(detections)
        else:
            self.stats['empty_frames'] += 1
        
        return detections
    
    def _postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Tuple]:
        if len(outputs) < 2:
            logger.warning("ONNX output has less than 2 tensors")
            return []
        
        boxes = outputs[0]
        logits = outputs[1]
        
        if boxes.ndim == 3:
            boxes = boxes[0]
        if logits.ndim == 3:
            logits = logits[0]
        
        logger.debug(f"Boxes shape after squeeze: {boxes.shape}")
        logger.debug(f"Logits shape after squeeze: {logits.shape}")
        
        scores = 1.0 / (1.0 + np.exp(-logits))
        
        if scores.ndim == 1:
            max_scores = scores
            class_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
        
        logger.debug(f"Max scores range: {max_scores.min():.4f} - {max_scores.max():.4f}")
        logger.debug(f"Scores > threshold ({self.confidence_threshold}): {np.sum(max_scores > self.confidence_threshold)}")
        
        mask = max_scores > self.confidence_threshold
        boxes_filtered = boxes[mask]
        scores_filtered = max_scores[mask]
        classes_filtered = class_ids[mask]
        
        orig_h, orig_w = original_shape
        detections = []
        for box, score, class_id in zip(boxes_filtered, scores_filtered, classes_filtered):
            x_center = box[0] * orig_w
            y_center = box[1] * orig_h
            width = box[2] * orig_w
            height = box[3] * orig_h
            
            detections.append((x_center, y_center, width, height, float(score), int(class_id)))
        
        logger.debug(f"Final detections: {len(detections)}")
        return detections
    
    def predict_ball_only(
        self, 
        frame: np.ndarray, 
        ball_class_id: int = 0, 
        use_temporal_filtering: bool = True,
        return_candidates: bool = False
    ) -> Optional[Tuple]:
        detections = self.predict(frame)
        ball_detections = [d for d in detections if d[5] == ball_class_id] if detections else []
        
        if not ball_detections:
            return (None, detections) if return_candidates else None
        
        best_detection = max(ball_detections, key=lambda x: x[4])[:5]
        return (best_detection, detections) if return_candidates else best_detection
    
    def get_stats(self) -> dict:
        """Get detector statistics"""
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        return {
            **self.stats,
            'avg_inference_time': avg_time,
            'current_fps': 1000 / avg_time if avg_time > 0 else 0
        }
