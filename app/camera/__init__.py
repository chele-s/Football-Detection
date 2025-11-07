"""
Módulo de Cámara Virtual para seguimiento inteligente de objetos.
Incluye algoritmos de suavizado y lógica de comportamiento de cámara.
"""

from .one_euro_filter import OneEuroFilter
from .virtual_camera import VirtualCamera

__all__ = ['OneEuroFilter', 'VirtualCamera']
