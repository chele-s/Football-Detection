"""
Cámara Virtual Inteligente para seguimiento de objetos.

Simula el comportamiento de un camarógrafo profesional:
- Dead-zones: No reacciona a micro-movimientos
- Anticipación: Predice hacia dónde va el objeto
- Límites seguros: No se sale del frame original
- Zoom adaptativo: Mantiene al objeto siempre visible
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from collections import deque
from .one_euro_filter import OneEuroFilter

logger = logging.getLogger(__name__)


class PIDController:
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, output_limits: Tuple[float, float] = (-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0
    
    def compute(self, error: float, dt: float = 1.0) -> float:
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        
        output_raw = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output_raw, *self.output_limits)
        
        if self.ki > 0:
            integral_correction = (output - output_raw) / self.ki
            self.integral += error * dt + integral_correction
            self.integral = np.clip(self.integral, -50.0, 50.0)
        else:
            self.integral += error * dt
            self.integral = np.clip(self.integral, -50.0, 50.0)
        
        self.last_error = error
        self.last_output = output
        return output
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_output = 0.0


class AdaptiveDeadZone:
    def __init__(self, base_size: float, velocity_scale: float = 0.5, max_size: float = 200.0):
        self.base_size = base_size
        self.velocity_scale = velocity_scale
        self.max_size = max_size
        self.velocity_history = deque(maxlen=10)
    
    def compute_threshold(self, velocity_magnitude: float) -> float:
        self.velocity_history.append(velocity_magnitude)
        avg_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        
        adaptive_size = self.base_size * (1.0 + self.velocity_scale * (avg_velocity / 100.0))
        adaptive_size = min(adaptive_size, self.max_size)
        
        return adaptive_size
    
    def should_move(self, distance: float, velocity_magnitude: float) -> bool:
        threshold = self.compute_threshold(velocity_magnitude)
        return distance > threshold * 0.7


class TrajectoryPredictor:
    def __init__(self, history_size: int = 30):
        self.position_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        self.acceleration_history = deque(maxlen=history_size)
    
    def update(self, position: np.ndarray, velocity: np.ndarray):
        self.position_history.append(position.copy())
        self.velocity_history.append(velocity.copy())
        
        if len(self.velocity_history) >= 2:
            accel = self.velocity_history[-1] - self.velocity_history[-2]
            self.acceleration_history.append(accel)
    
    def predict(self, steps_ahead: int = 5) -> np.ndarray:
        if len(self.position_history) < 3:
            return self.position_history[-1] if self.position_history else np.zeros(2)
        
        positions = np.array(list(self.position_history)[-10:])
        
        if len(positions) >= 3:
            coeffs_x = np.polyfit(np.arange(len(positions)), positions[:, 0], deg=2)
            coeffs_y = np.polyfit(np.arange(len(positions)), positions[:, 1], deg=2)
            
            future_idx = len(positions) + steps_ahead
            pred_x = np.polyval(coeffs_x, future_idx)
            pred_y = np.polyval(coeffs_y, future_idx)
            
            return np.array([pred_x, pred_y])
        
        return positions[-1]
    
    def get_curvature(self) -> float:
        if len(self.position_history) < 3:
            return 0.0
        
        positions = np.array(list(self.position_history)[-5:])
        if len(positions) < 3:
            return 0.0
        
        try:
            dx = np.gradient(positions[:, 0])
            dy = np.gradient(positions[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            denom = (dx**2 + dy**2)**1.5
            denom = np.where(denom < 1e-9, 1e-9, denom)
            curvature = np.abs(dx * ddy - dy * ddx) / denom
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
            return np.mean(curvature[-3:])
        except:
            return 0.0


class VirtualCamera:
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        output_width: int = 1920,
        output_height: int = 1080,
        dead_zone_percent: float = 0.10,
        anticipation_factor: float = 0.3,
        zoom_padding: float = 1.2,
        smoothing_freq: float = 30.0,
        smoothing_min_cutoff: float = 1.0,
        smoothing_beta: float = 0.007,
        use_pid: bool = True,
        prediction_steps: int = 5
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.output_width = output_width
        self.output_height = output_height
        self.use_pid = use_pid
        self.prediction_steps = prediction_steps
        
        self.effective_width = int(output_width * zoom_padding)
        self.effective_height = int(output_height * zoom_padding)
        
        if self.effective_width > frame_width:
            self.effective_width = frame_width
        if self.effective_height > frame_height:
            self.effective_height = frame_height
        
        self.base_dead_zone = dead_zone_percent * output_width
        self.anticipation_factor = anticipation_factor
        self.zoom_padding = zoom_padding
        
        self.position_filter = OneEuroFilter(
            freq=smoothing_freq,
            min_cutoff=smoothing_min_cutoff,
            beta=smoothing_beta,
            adaptive_beta=True
        )
        
        self.adaptive_deadzone = AdaptiveDeadZone(
            base_size=self.base_dead_zone,
            velocity_scale=0.5,
            max_size=output_width * 0.3
        )
        
        self.trajectory_predictor = TrajectoryPredictor(history_size=30)
        
        if use_pid:
            self.pid_x = PIDController(kp=0.8, ki=0.01, kd=0.15, output_limits=(-500, 500))
            self.pid_y = PIDController(kp=0.8, ki=0.01, kd=0.15, output_limits=(-500, 500))
        
        self.current_center_x = frame_width // 2
        self.current_center_y = frame_height // 2
        self.last_target = np.array([frame_width / 2, frame_height / 2])
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        
        self.position_history = deque(maxlen=60)
        self.velocity_history = deque(maxlen=30)
        self.frame_count = 0
        
        self.stats = {
            'total_updates': 0,
            'deadzone_blocks': 0,
            'pid_corrections': 0,
            'predictions_used': 0,
            'zoom_adjustments': 0
        }
        
        logger.info(f"VirtualCamera initialized: {frame_width}x{frame_height} -> {output_width}x{output_height}")
    
    def update(
        self,
        target_x: float,
        target_y: float,
        timestamp: Optional[float] = None,
        velocity_hint: Optional[Tuple[float, float]] = None
    ) -> Tuple[int, int, int, int]:
        self.stats['total_updates'] += 1
        self.frame_count += 1
        
        current_target = np.array([target_x, target_y])
        current_velocity = current_target - self.last_target
        
        if velocity_hint is not None:
            current_velocity = np.array(velocity_hint)
        
        self.velocity = 0.7 * self.velocity + 0.3 * current_velocity
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        self.trajectory_predictor.update(current_target, self.velocity)
        predicted_position = self.trajectory_predictor.predict(self.prediction_steps)
        
        curvature = self.trajectory_predictor.get_curvature()
        if curvature > 0.01:
            self.stats['predictions_used'] += 1
            anticipated_target = predicted_position
            logger.debug(f"High curvature detected: {curvature:.4f}, using prediction")
        else:
            anticipation_scale = self.anticipation_factor * (1.0 + min(velocity_magnitude / 100.0, 1.0))
            anticipated_target = current_target + self.velocity * anticipation_scale
        
        current_camera_pos = np.array([self.current_center_x, self.current_center_y])
        error = anticipated_target - current_camera_pos
        distance = np.linalg.norm(error)
        
        should_move_x = self.adaptive_deadzone.should_move(abs(error[0]), velocity_magnitude)
        should_move_y = self.adaptive_deadzone.should_move(abs(error[1]), velocity_magnitude)
        
        if not should_move_x:
            self.stats['deadzone_blocks'] += 1
            desired_x = self.current_center_x
        else:
            if self.use_pid:
                self.stats['pid_corrections'] += 1
                correction_x = self.pid_x.compute(error[0], dt=1.0/30.0)
                desired_x = self.current_center_x + correction_x
            else:
                desired_x = anticipated_target[0]
        
        if not should_move_y:
            desired_y = self.current_center_y
        else:
            if self.use_pid:
                correction_y = self.pid_y.compute(error[1], dt=1.0/30.0)
                desired_y = self.current_center_y + correction_y
            else:
                desired_y = anticipated_target[1]
        
        smooth_x, smooth_y = self.position_filter(
            desired_x,
            desired_y,
            timestamp
        )
        
        self.current_center_x = smooth_x
        self.current_center_y = smooth_y
        self.last_target = current_target
        
        self.position_history.append(np.array([smooth_x, smooth_y]))
        self.velocity_history.append(self.velocity.copy())
        
        # Disable dynamic zoom for tight, fixed zoom on ball
        # dynamic_zoom = self._compute_dynamic_zoom(velocity_magnitude, curvature)
        # if abs(dynamic_zoom - self.zoom_padding) > 0.05:
        #     self.set_zoom(dynamic_zoom)
        #     self.stats['zoom_adjustments'] += 1
        
        crop = self._calculate_crop(smooth_x, smooth_y)
        
        if self.frame_count % 100 == 0:
            logger.debug(f"Camera stats: updates={self.stats['total_updates']}, "
                        f"deadzone_blocks={self.stats['deadzone_blocks']}, "
                        f"pid_corrections={self.stats['pid_corrections']}, "
                        f"vel_mag={velocity_magnitude:.2f}")
        
        return crop
    
    def _compute_dynamic_zoom(self, velocity_magnitude: float, curvature: float) -> float:
        base_zoom = self.zoom_padding
        
        velocity_factor = min(velocity_magnitude / 200.0, 0.3)
        curvature_factor = min(curvature * 10.0, 0.2)
        
        dynamic_zoom = base_zoom * (1.0 + velocity_factor + curvature_factor)
        dynamic_zoom = np.clip(dynamic_zoom, 1.0, 2.0)
        
        return dynamic_zoom
    
    def _calculate_crop(self, center_x: float, center_y: float) -> Tuple[int, int, int, int]:
        half_w = self.effective_width / 2
        half_h = self.effective_height / 2
        
        x1 = int(center_x - half_w)
        y1 = int(center_y - half_h)
        x2 = int(center_x + half_w)
        y2 = int(center_y + half_h)
        
        margin = 10
        if x1 < margin:
            x1 = margin
            x2 = margin + self.effective_width
        if x2 > self.frame_width - margin:
            x2 = self.frame_width - margin
            x1 = x2 - self.effective_width
        
        if y1 < margin:
            y1 = margin
            y2 = margin + self.effective_height
        if y2 > self.frame_height - margin:
            y2 = self.frame_height - margin
            y1 = y2 - self.effective_height
        
        x1 = max(0, min(x1, self.frame_width - self.effective_width))
        y1 = max(0, min(y1, self.frame_height - self.effective_height))
        x2 = x1 + self.effective_width
        y2 = y1 + self.effective_height
        
        return (x1, y1, x2, y2)
    
    def get_current_crop(self) -> Tuple[int, int, int, int]:
        """Retorna el crop actual sin actualizar."""
        return self._calculate_crop(self.current_center_x, self.current_center_y)
    
    def get_stats(self) -> dict:
        return self.stats.copy()
    
    def get_current_velocity(self) -> Tuple[float, float]:
        return tuple(self.velocity)
    
    def reset(self, safe_search_position: Optional[Tuple[float, float]] = None):
        logger.info("Resetting Virtual Camera")
        if safe_search_position:
            self.current_center_x = safe_search_position[0]
            self.current_center_y = safe_search_position[1]
            self.last_target = np.array([safe_search_position[0], safe_search_position[1]])
        else:
            self.current_center_x = int(self.frame_width * 0.5)
            self.current_center_y = int(self.frame_height * 0.48)
            self.last_target = np.array([self.frame_width * 0.5, self.frame_height * 0.48])
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.position_filter.reset()
        self.trajectory_predictor = TrajectoryPredictor(history_size=30)
        if self.use_pid:
            self.pid_x.reset()
            self.pid_y.reset()
        self.position_history.clear()
        self.velocity_history.clear()
        self.frame_count = 0
        self.stats = {
            'total_updates': 0,
            'deadzone_blocks': 0,
            'pid_corrections': 0,
            'predictions_used': 0,
            'zoom_adjustments': 0
        }
    
    def set_zoom(self, zoom_padding: float):
        zoom_padding = np.clip(zoom_padding, 1.0, 2.0)
        self.zoom_padding = zoom_padding
        self.effective_width = int(self.output_width * zoom_padding)
        self.effective_height = int(self.output_height * zoom_padding)
        
        if self.effective_width > self.frame_width:
            self.effective_width = self.frame_width
        if self.effective_height > self.frame_height:
            self.effective_height = self.frame_height
        
        logger.debug(f"Zoom adjusted to {zoom_padding:.2f} -> {self.effective_width}x{self.effective_height}")
