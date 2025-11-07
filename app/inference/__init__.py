"""
Módulo de Inferencia de IA para detección de objetos.
Wrapper optimizado para modelos YOLOv8 con soporte GPU/TensorRT.
"""

from .detector import BallDetector

__all__ = ['BallDetector']
