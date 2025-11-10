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
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        provider = self.session.get_providers()[0]
        logger.info(f"✓ ONNX Runtime session created")
        logger.info(f"✓ Using provider: {provider}")
        logger.info(f"✓ Input: {self.input_name}")
        logger.info(f"✓ Outputs: {self.output_names}")
        
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
        """Run inference on frame"""
        start_time = time.time()
        self.stats['total_inferences'] += 1
        
        # Preprocess
        input_data = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )
        
        # Parse outputs (RF-DETR outputs boxes and labels)
        # Format depends on ONNX export, typically:
        # outputs[0] = boxes [N, 4]
        # outputs[1] = labels/scores [N, num_classes]
        detections = self._postprocess(outputs)
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        if detections:
            self.stats['total_detections'] += len(detections)
        else:
            self.stats['empty_frames'] += 1
        
        return detections
    
    def _postprocess(self, outputs: List[np.ndarray]) -> List[Tuple]:
        if len(outputs) < 2:
            return []
        
        boxes, logits = outputs[0][0], outputs[1][0]
        scores = 1.0 / (1.0 + np.exp(-logits))
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        mask = max_scores > self.confidence_threshold
        boxes_filtered = boxes[mask]
        scores_filtered = max_scores[mask]
        classes_filtered = class_ids[mask]
        
        scale = self.imgsz
        return [
            (
                box[0] * scale,
                box[1] * scale,
                box[2] * scale,
                box[3] * scale,
                float(score),
                int(class_id)
            )
            for box, score, class_id in zip(boxes_filtered, scores_filtered, classes_filtered)
        ]
    
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
