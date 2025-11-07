"""
Implementación del filtro One-Euro para suavizado adaptativo de coordenadas.

El algoritmo One-Euro es ideal para tracking en tiempo real porque:
- Reduce el ruido cuando el objeto se mueve lentamente
- Responde rápidamente cuando el objeto se mueve rápido
- Tiene latencia ultra-baja (crítico para <33ms por frame)

Paper original: http://cristal.univ-lille.fr/~casiez/1euro/
"""

import math
import logging
import numpy as np
from typing import Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveLowPassFilter:
    def __init__(self, alpha: float, outlier_threshold: float = 3.0):
        self.alpha = alpha
        self.outlier_threshold = outlier_threshold
        self.y = None
        self.s = None
        self.velocity_history = deque(maxlen=10)
        self.value_history = deque(maxlen=30)
    
    def filter(self, x: float, alpha: Optional[float] = None) -> float:
        if alpha is None:
            alpha = self.alpha
        
        alpha = np.clip(alpha, 0.0, 1.0)
        
        if self.y is None:
            self.s = x
            self.y = x
            self.value_history.append(x)
            return self.s
        
        if self._is_outlier(x):
            logger.debug(f"Outlier detectado: {x:.2f}, usando predicción")
            x = self._predict_value()
        
        self.s = alpha * x + (1 - alpha) * self.s
        self.y = self.s
        self.value_history.append(self.s)
        
        return self.s
    
    def _is_outlier(self, x: float) -> bool:
        if len(self.value_history) < 5:
            return False
        
        recent = list(self.value_history)[-10:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        if std < 1e-6:
            return False
        
        z_score = abs((x - mean) / std)
        return z_score > self.outlier_threshold
    
    def _predict_value(self) -> float:
        if len(self.value_history) < 2:
            return self.s
        
        recent = list(self.value_history)[-5:]
        if len(recent) >= 2:
            velocity = (recent[-1] - recent[-2])
            return recent[-1] + velocity
        
        return self.s


class OneEuroFilter:
    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        outlier_threshold: float = 3.0,
        adaptive_beta: bool = True
    ):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.adaptive_beta = adaptive_beta
        self.outlier_threshold = outlier_threshold
        
        self.x_filter = AdaptiveLowPassFilter(self._alpha(min_cutoff), outlier_threshold)
        self.dx_filter = AdaptiveLowPassFilter(self._alpha(d_cutoff), outlier_threshold)
        
        self.y_filter = AdaptiveLowPassFilter(self._alpha(min_cutoff), outlier_threshold)
        self.dy_filter = AdaptiveLowPassFilter(self._alpha(d_cutoff), outlier_threshold)
        
        self.last_time = None
        self.velocity_history = deque(maxlen=20)
        self.jerk_history = deque(maxlen=10)
        self.last_velocity = np.array([0.0, 0.0])
        
        self.stats = {
            'total_calls': 0,
            'outliers_detected': 0,
            'beta_adjustments': 0
        }
    
    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)
    
    def __call__(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        self.stats['total_calls'] += 1
        
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 1e-6:
                self.freq = 1.0 / dt
        
        self.last_time = timestamp
        
        dx = 0.0 if self.x_filter.y is None else (x - self.x_filter.y) * self.freq
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff))
        
        dy = 0.0 if self.y_filter.y is None else (y - self.y_filter.y) * self.freq
        edy = self.dy_filter.filter(dy, self._alpha(self.d_cutoff))
        
        velocity_magnitude = math.sqrt(edx**2 + edy**2)
        self.velocity_history.append(velocity_magnitude)
        
        current_velocity = np.array([edx, edy])
        jerk = np.linalg.norm(current_velocity - self.last_velocity) * self.freq
        self.jerk_history.append(jerk)
        self.last_velocity = current_velocity
        
        effective_beta = self._compute_adaptive_beta() if self.adaptive_beta else self.beta
        
        cutoff_x = self.min_cutoff + effective_beta * abs(edx)
        cutoff_y = self.min_cutoff + effective_beta * abs(edy)
        
        cutoff_x = np.clip(cutoff_x, self.min_cutoff, 10.0)
        cutoff_y = np.clip(cutoff_y, self.min_cutoff, 10.0)
        
        x_filtered = self.x_filter.filter(x, self._alpha(cutoff_x))
        y_filtered = self.y_filter.filter(y, self._alpha(cutoff_y))
        
        if self.stats['total_calls'] % 100 == 0:
            logger.debug(f"Filter stats: calls={self.stats['total_calls']}, "
                        f"outliers={self.stats['outliers_detected']}, "
                        f"beta_adj={self.stats['beta_adjustments']}, "
                        f"vel_mag={velocity_magnitude:.2f}")
        
        return x_filtered, y_filtered
    
    def _compute_adaptive_beta(self) -> float:
        if len(self.velocity_history) < 5:
            return self.beta
        
        recent_velocities = list(self.velocity_history)[-10:]
        avg_velocity = np.mean(recent_velocities)
        velocity_std = np.std(recent_velocities)
        
        if len(self.jerk_history) >= 3:
            avg_jerk = np.mean(list(self.jerk_history)[-5:])
            jerk_factor = np.clip(avg_jerk / 1000.0, 0.5, 2.0)
        else:
            jerk_factor = 1.0
        
        velocity_factor = 1.0 + (avg_velocity / 100.0)
        stability_factor = 1.0 / (1.0 + velocity_std)
        
        adaptive_beta = self.beta * velocity_factor * jerk_factor * stability_factor
        adaptive_beta = np.clip(adaptive_beta, self.beta * 0.5, self.beta * 3.0)
        
        if abs(adaptive_beta - self.beta) > 0.001:
            self.stats['beta_adjustments'] += 1
        
        return adaptive_beta
    
    def get_velocity(self) -> Tuple[float, float]:
        return tuple(self.last_velocity)
    
    def get_stats(self) -> dict:
        return self.stats.copy()
    
    def reset(self):
        logger.info("Resetting One-Euro Filter")
        self.x_filter = AdaptiveLowPassFilter(self._alpha(self.min_cutoff), self.outlier_threshold)
        self.dx_filter = AdaptiveLowPassFilter(self._alpha(self.d_cutoff), self.outlier_threshold)
        self.y_filter = AdaptiveLowPassFilter(self._alpha(self.min_cutoff), self.outlier_threshold)
        self.dy_filter = AdaptiveLowPassFilter(self._alpha(self.d_cutoff), self.outlier_threshold)
        self.last_time = None
        self.velocity_history.clear()
        self.jerk_history.clear()
        self.last_velocity = np.array([0.0, 0.0])
        self.stats = {'total_calls': 0, 'outliers_detected': 0, 'beta_adjustments': 0}
