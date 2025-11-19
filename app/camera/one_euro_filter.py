"""
Implementación robusta y optimizada del filtro One-Euro.

Referencia: http://cristal.univ-lille.fr/~casiez/1euro/
"""

import math
import time
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class LowPassFilter:
    """
    Filtro paso bajo exponencial simple.
    x_i = alpha * x_i + (1 - alpha) * x_{i-1}
    """
    __slots__ = ('alpha', 'y', 's')

    def __init__(self, alpha: float, init_value: float = 0.0):
        self.alpha = alpha
        self.y = init_value
        self.s = init_value
        self.initialized = False

    def filter(self, value: float, alpha: float = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        
        if not self.initialized:
            self.s = value
            self.y = value
            self.initialized = True
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
            self.y = self.s
        return self.y

    def filter_with_alpha(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.s = value
            self.y = value
            self.initialized = True
        else:
            self.s = alpha * value + (1.0 - alpha) * self.s
            self.y = self.s
        return self.y

    def reset(self):
        self.initialized = False


class OneEuroFilter:
    """
    Filtro One-Euro optimizado para tracking en tiempo real.
    Minimiza el jitter (temblores) en bajas velocidades y el lag en altas velocidades.
    """
    def __init__(
        self,
        freq: float = 30.0,     # Frecuencia estimada de entrada
        min_cutoff: float = 1.0, # Frecuencia de corte mínima (para estados estables)
        beta: float = 0.007,     # Coeficiente de velocidad (sensibilidad al movimiento)
        d_cutoff: float = 1.0    # Frecuencia de corte para la derivada
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(self.d_cutoff))
        self.y_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self.dy_filter = LowPassFilter(self._alpha(self.d_cutoff))
        
        self.last_time = None
        self.last_x = None
        self.last_y = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        # Actualizar frecuencia si tenemos timestamps
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt > 0:
                self.freq = 1.0 / dt
        self.last_time = timestamp

        # Estimar derivada (velocidad)
        if self.x_filter.initialized:
            dx_raw = (x - self.x_filter.y) * self.freq
            dy_raw = (y - self.y_filter.y) * self.freq
        else:
            dx_raw = 0.0
            dy_raw = 0.0

        # Filtrar derivada
        edx = self.dx_filter.filter_with_alpha(dx_raw, self._alpha(self.d_cutoff))
        edy = self.dy_filter.filter_with_alpha(dy_raw, self._alpha(self.d_cutoff))

        # Calcular cutoff adaptativo basado en la velocidad
        # cutoff = min_cutoff + beta * |edx|
        # A mayor velocidad, mayor cutoff (menos filtrado, menos lag)
        # A menor velocidad, menor cutoff (más filtrado, menos jitter)
        cutoff_x = self.min_cutoff + self.beta * abs(edx)
        cutoff_y = self.min_cutoff + self.beta * abs(edy)

        # Filtrar señal principal
        x_filtered = self.x_filter.filter_with_alpha(x, self._alpha(cutoff_x))
        y_filtered = self.y_filter.filter_with_alpha(y, self._alpha(cutoff_y))

        return x_filtered, y_filtered

    def set_smoothing_level(self, level: float):
        """Ajusta dinámicamente el nivel de suavizado (min_cutoff)."""
        # level bajo (0.1) = mucho suavizado (lento)
        # level alto (5.0) = poco suavizado (rápido)
        self.min_cutoff = max(0.01, float(level))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.y_filter.reset()
        self.dy_filter.reset()
        self.last_time = None
