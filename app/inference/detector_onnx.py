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
        """
        Post-process ONNX outputs to standard format.
        
        Returns list of (x_center, y_center, width, height, score, class_id)
        """
        detections = []
        
        # RF-DETR ONNX export typically outputs:
        # outputs[0] = pred_boxes [1, num_queries, 4] (normalized x,y,w,h)
        # outputs[1] = pred_logits [1, num_queries, num_classes]
        
        if len(outputs) >= 2:
            boxes = outputs[0][0]  # Remove batch dim
            logits = outputs[1][0]  # Remove batch dim
            
            # Get scores (apply sigmoid or softmax depending on model)
            scores = 1 / (1 + np.exp(-logits))  # Sigmoid
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
            
            # Filter by confidence
            mask = max_scores > self.confidence_threshold
            
            for box, score, class_id in zip(boxes[mask], max_scores[mask], class_ids[mask]):
                # box is [x_center, y_center, width, height] normalized [0, 1]
                # Scale to image size
                x_center = box[0] * self.imgsz
                y_center = box[1] * self.imgsz
                width = box[2] * self.imgsz
                height = box[3] * self.imgsz
                
                detections.append((x_center, y_center, width, height, float(score), int(class_id)))
        
        return detections
    
    def predict_ball_only(self, frame: np.ndarray, ball_class_id: int = 0, roi: Optional[Tuple] = None, **kwargs) -> Optional[Tuple]:
        """Get highest confidence ball detection"""
        # Note: ROI cropping not implemented in ONNX detector yet
        # Full frame inference is used for simplicity
        detections = self.predict(frame, **kwargs)
        
        if not detections:
            return None
        
        # Filter by class_id if specified
        if ball_class_id is not None:
            detections = [d for d in detections if d[5] == ball_class_id]
        
        if not detections:
            return None
        
        # Return the highest confidence detection
        return max(detections, key=lambda x: x[4])
    
    def get_stats(self) -> dict:
        """Get detector statistics"""
        avg_time = np.mean(self.inference_times) if self.inference_times else 0
        return {
            **self.stats,
            'avg_inference_time': avg_time,
            'current_fps': 1000 / avg_time if avg_time > 0 else 0
        }
