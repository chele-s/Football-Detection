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
    
    # ImageNet normalization constants used by RF-DETR
    MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        onnx_path: str,
        confidence_threshold: float = 0.12,
        imgsz: int = 640,  # Changed default to 640 to match PyTorch version
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
    
    def _resize_keeping_aspect_ratio(self, image: np.ndarray, desired_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image preserving aspect ratio.
        
        Args:
            image: numpy array representing the image
            desired_size: tuple (width, height) representing target dimensions
        
        Returns:
            Resized image
        """
        img_ratio = image.shape[1] / image.shape[0]  # width / height
        desired_ratio = desired_size[0] / desired_size[1]
        
        # Determine new dimensions
        if img_ratio >= desired_ratio:
            # Resize by width
            new_width = desired_size[0]
            new_height = int(desired_size[0] / img_ratio)
        else:
            # Resize by height
            new_height = desired_size[1]
            new_width = int(desired_size[1] * img_ratio)
        
        return cv2.resize(image, (new_width, new_height))
    
    def _letterbox(self, image: np.ndarray, desired_size: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
        """
        Resize and pad image to fit desired size, preserving aspect ratio.
        Uses letterbox method (adds borders) instead of stretching.
        
        Args:
            image: numpy array representing the image
            desired_size: tuple (width, height) representing target dimensions
            color: tuple (B, G, R) representing the color to pad with
        
        Returns:
            Letterboxed image
        """
        # Resize keeping aspect ratio
        resized_img = self._resize_keeping_aspect_ratio(image, desired_size)
        new_height, new_width = resized_img.shape[:2]
        
        # Calculate padding
        top_padding = (desired_size[1] - new_height) // 2
        bottom_padding = desired_size[1] - new_height - top_padding
        left_padding = (desired_size[0] - new_width) // 2
        right_padding = desired_size[0] - new_width - left_padding
        
        # Store padding info for postprocessing
        self._pad_x = left_padding
        self._pad_y = top_padding
        self._scale = min(desired_size[0] / image.shape[1], desired_size[1] / image.shape[0])
        
        # Add border padding
        return cv2.copyMakeBorder(
            resized_img,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            cv2.BORDER_CONSTANT,
            value=color
        )
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        self._original_h, self._original_w = frame.shape[:2]
        
        # Apply letterbox (resize keeping aspect ratio + padding)
        frame_letterboxed = self._letterbox(frame, (self.imgsz, self.imgsz), color=(114, 114, 114))
        
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame_letterboxed, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization: (image - mean) / std
        # Broadcasting: subtract mean and divide by std for each channel
        frame_normalized = (frame_normalized - self.MEANS) / self.STDS
        
        # Convert to CHW format (Channel, Height, Width)
        frame_chw = frame_normalized.transpose(2, 0, 1)
        
        # Add batch dimension: (1, C, H, W)
        return np.expand_dims(frame_chw, axis=0).astype(np.float32)
    
    def _postprocess(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Tuple]:
        """
        Post-process model outputs following RF-DETR official post-processing.
        Handles letterbox padding to return coordinates in original image space.
        """
        if len(outputs) < 2:
            logger.warning("ONNX output has less than 2 tensors")
            return []
        
        boxes = outputs[0]  # Format: [batch, num_queries, 4] in cxcywh normalized [0,1]
        logits = outputs[1]  # Format: [batch, num_queries, num_classes]
        
        # Remove batch dimension
        if boxes.ndim == 3:
            boxes = boxes[0]
        if logits.ndim == 3:
            logits = logits[0]
        
        logger.debug(f"Boxes shape after squeeze: {boxes.shape}")
        logger.debug(f"Logits shape after squeeze: {logits.shape}")
        
        # Apply sigmoid to get probabilities
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        
        # Get max score and class per detection
        if scores.ndim == 1:
            max_scores = scores
            class_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)
        
        logger.debug(f"Max scores range: {max_scores.min():.4f} - {max_scores.max():.4f}")
        logger.debug(f"Scores > threshold ({self.confidence_threshold}): {np.sum(max_scores > self.confidence_threshold)}")
        
        # Filter by confidence threshold
        mask = max_scores > self.confidence_threshold
        boxes_filtered = boxes[mask]
        scores_filtered = max_scores[mask]
        classes_filtered = class_ids[mask]
        
        orig_h, orig_w = original_shape
        detections = []
        
        for i, (box, score, class_id) in enumerate(zip(boxes_filtered, scores_filtered, classes_filtered)):
            logger.debug(f"Detection {i}: box_raw={box}, score={score:.4f}, class={class_id}")
            
            # RF-DETR outputs boxes in cxcywh format normalized to [0, 1]
            # Convert to xyxy format first
            cx_norm, cy_norm, w_norm, h_norm = box[0], box[1], box[2], box[3]
            x1_norm = cx_norm - 0.5 * w_norm
            y1_norm = cy_norm - 0.5 * h_norm
            x2_norm = cx_norm + 0.5 * w_norm
            y2_norm = cy_norm + 0.5 * h_norm
            
            # Scale to input image size (with letterbox padding)
            x1_input = x1_norm * self.imgsz
            y1_input = y1_norm * self.imgsz
            x2_input = x2_norm * self.imgsz
            y2_input = y2_norm * self.imgsz
            
            # Remove letterbox padding
            x1_input -= self._pad_x
            y1_input -= self._pad_y
            x2_input -= self._pad_x
            y2_input -= self._pad_y
            
            # Scale back to original image size
            x1_orig = x1_input / self._scale
            y1_orig = y1_input / self._scale
            x2_orig = x2_input / self._scale
            y2_orig = y2_input / self._scale
            
            # Clip to image bounds
            x1_orig = np.clip(x1_orig, 0, orig_w)
            y1_orig = np.clip(y1_orig, 0, orig_h)
            x2_orig = np.clip(x2_orig, 0, orig_w)
            y2_orig = np.clip(y2_orig, 0, orig_h)
            
            # Convert back to cxcywh format for compatibility
            x_center = (x1_orig + x2_orig) / 2
            y_center = (y1_orig + y2_orig) / 2
            width = x2_orig - x1_orig
            height = y2_orig - y1_orig
            
            logger.debug(f"  → Final: cx={x_center:.1f}, cy={y_center:.1f}, w={width:.1f}, h={height:.1f}")
            
            if width > 0 and height > 0:
                detections.append((x_center, y_center, width, height, float(score), int(class_id)))
            else:
                logger.warning(f"  → SKIPPED: Invalid dimensions w={width} h={height}")
        
        logger.info(f"✓ Returning {len(detections)} valid detections")
        if detections:
            logger.info(f"  Best detection: {detections[0]}")
        return detections
    
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
