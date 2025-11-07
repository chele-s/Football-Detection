"""
Módulo de Tracking para seguimiento persistente de objetos.
Integra ByteTrack con filtro de Kalman para robustez ante oclusiones.
"""

from .tracker import BallTracker

__all__ = ['BallTracker']
