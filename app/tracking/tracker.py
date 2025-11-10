import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter:
    def __init__(self, dt: float = 1/30, process_noise: float = 0.01, measurement_noise: float = 5.0):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.base_measurement_noise = measurement_noise
        
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float64)
        
        self.Q = np.eye(6, dtype=np.float64) * process_noise
        self.Q[4:, 4:] *= 2.0
        
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.initialized = False
        
        self.innovation_history = deque(maxlen=10)
        self.mahalanobis_history = deque(maxlen=10)
    
    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.P = (self.P + self.P.T) / 2
        
        return float(self.x[0, 0]), float(self.x[1, 0])
    
    def update(self, z: np.ndarray) -> float:
        y = z - self.H @ self.x
        self.innovation_history.append(np.linalg.norm(y))
        
        S = self.H @ self.P @ self.H.T + self.R
        S = (S + S.T) / 2
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, adding regularization")
            S += np.eye(2) * 1e-4
            S_inv = np.linalg.inv(S)
        
        mahalanobis = float(y.T @ S_inv @ y)
        self.mahalanobis_history.append(mahalanobis)
        
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        self.P = (self.P + self.P.T) / 2
        
        self.initialized = True
        return mahalanobis
    
    def correct(self, x: float, y: float) -> float:
        z = np.array([[x], [y]], dtype=np.float64)
        if not self.initialized:
            self.x[0, 0] = x
            self.x[1, 0] = y
            self.initialized = True
            return 0.0
        else:
            return self.update(z)
    
    def get_velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])
    
    def get_acceleration(self) -> Tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])
    
    def get_state_vector(self) -> np.ndarray:
        return self.x.flatten()
    
    def is_stable(self) -> bool:
        if len(self.innovation_history) < 5:
            return True
        
        recent_innovations = list(self.innovation_history)[-5:]
        return np.std(recent_innovations) < 50.0
    
    def adapt_noise(self, factor: float = 1.0):
        self.Q *= factor
        self.R *= factor
        logger.debug(f"Kalman noise adapted by factor {factor:.2f}")
    
    def set_measurement_noise(self, noise: float):
        self.measurement_noise = noise
        self.R = np.eye(2, dtype=np.float64) * noise


class BallTracker:
    def __init__(
        self,
        max_lost_frames: int = 10,
        min_confidence: float = 0.3,
        history_size: int = 30,
        dt: float = 1/30,
        iou_threshold: float = 0.3,
        adaptive_noise: bool = True
    ):
        self.max_lost_frames = max_lost_frames
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.iou_threshold = iou_threshold
        self.adaptive_noise = adaptive_noise
        
        self.kalman = ExtendedKalmanFilter(dt=dt, process_noise=0.01, measurement_noise=5.0)
        self.lost_frames = 0
        self.track_id = 0
        self.is_tracking = False
        self.consecutive_detections = 0
        
        self.position_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        self.mahalanobis_history = deque(maxlen=20)
        self.movement_history = deque(maxlen=15)
        
        self.last_detection = None
        self.predicted_position = None
        self.last_bbox = None
        
        self.hypotheses = []
        self.hypothesis_scores = []
        
        self.stats = {
            'total_updates': 0,
            'successful_tracks': 0,
            'predictions_used': 0,
            'noise_adaptations': 0,
            'outliers_rejected': 0,
            'erratic_detections': 0
        }
        
        self.stability_lock = 0
        self.stability_lock_max = 20
        
        logger.info(f"BallTracker initialized: max_lost={max_lost_frames}, min_conf={min_confidence}")
    
    def update(
        self,
        detection: Optional[Tuple[float, float, float, float, float]],
        detections_list: Optional[List[Tuple[float, float, float, float, float]]] = None
    ) -> Optional[Tuple[float, float, bool]]:
        self.stats['total_updates'] += 1
        
        tracking_strength = min(self.consecutive_detections / 15.0, 1.0)
        is_stable = tracking_strength > 0.5
        is_recovering = self.lost_frames > 0 and self.lost_frames <= 5
        
        detector_erratic = self._is_detector_erratic()
        movement_variance = self._calculate_movement_variance()
        is_very_stable = self.consecutive_detections >= 25 and movement_variance < 15.0
        
        if self.stability_lock > 0:
            self.stability_lock -= 1
        
        if detection is not None:
            x_center, y_center, width, height, confidence = detection
            
            if self.stability_lock > 0 and self.kalman.initialized:
                pred_x, pred_y = float(self.kalman.x[0, 0]), float(self.kalman.x[1, 0])
                dist_to_pred = float(np.sqrt((x_center - pred_x)**2 + (y_center - pred_y)**2))
                lock_threshold = 80.0
                
                if dist_to_pred > lock_threshold:
                    logger.debug(f"Stability lock active: ignoring detection {dist_to_pred:.1f}px away")
                    detection = None
            
            if detection is not None and confidence < self.min_confidence:
                if is_stable:
                    logger.debug(f"Low confidence {confidence:.2f}, but tracker stable - accepting")
                else:
                    detection = None
        
        if detection is not None:
            x_center, y_center, width, height, confidence = detection
            bbox = (x_center, y_center, width, height)
            
            if not self.kalman.initialized:
                _ = self.kalman.correct(x_center, y_center)
                vx, vy = self.kalman.get_velocity()
                self.velocity_history.append((vx, vy))
                self.lost_frames = 0
                self.consecutive_detections = 1
                self.is_tracking = True
                self.last_detection = detection
                self.last_bbox = bbox
                self.position_history.append((x_center, y_center))
                self.confidence_history.append(confidence)
                self.stats['successful_tracks'] += 1
                self.kalman.set_measurement_noise(5.0)
                return (x_center, y_center, True)
            
            should_gate = is_stable and not is_recovering
            
            if detector_erratic and is_stable:
                self.stats['erratic_detections'] += 1
                logger.debug(f"Detector erratic (variance={movement_variance:.1f}), stricter gating")
                should_gate = True
            
            if should_gate and self.last_bbox is not None:
                vx_est, vy_est = self.get_velocity()
                vmag = float(np.sqrt(vx_est**2 + vy_est**2))
                
                base_distance = 180.0
                velocity_factor = min(vmag * 2.5, 800.0)
                lost_relaxation = min(self.lost_frames * 150.0, 600.0)
                erratic_penalty = 80.0 if detector_erratic else 0.0
                allowed_distance = base_distance + velocity_factor + lost_relaxation - erratic_penalty
                
                dist_curr = float(np.sqrt((x_center - float(self.kalman.x[0,0]))**2 + (y_center - float(self.kalman.x[1,0]))**2))
                
                if dist_curr > allowed_distance:
                    if confidence > 0.7:
                        logger.info(f"High confidence detection ({confidence:.2f}) overrides distance gate ({dist_curr:.1f} > {allowed_distance:.1f})")
                    else:
                        logger.debug(f"Distance gate: {dist_curr:.1f} > {allowed_distance:.1f}")
                        self.stats['outliers_rejected'] += 1
                        detection = None
                
                if detection is not None:
                    H = self.kalman.H
                    P = self.kalman.P
                    R = self.kalman.R
                    xhat = self.kalman.x
                    z = np.array([[x_center],[y_center]], dtype=np.float64)
                    y = z - H @ xhat
                    S = H @ P @ H.T + R
                    S = (S + S.T) / 2
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S += np.eye(2) * 1e-4
                        S_inv = np.linalg.inv(S)
                    maha = float(y.T @ S_inv @ y)
                    
                    base_threshold = 15.0
                    lost_bonus = min(self.lost_frames * 8.0, 40.0)
                    confidence_bonus = max(0, (confidence - 0.6) * 25.0)
                    erratic_penalty = 8.0 if detector_erratic else 0.0
                    gate_threshold = base_threshold + lost_bonus + confidence_bonus - erratic_penalty
                    
                    if maha > gate_threshold:
                        if confidence > 0.75:
                            logger.info(f"High confidence ({confidence:.2f}) overrides Mahalanobis gate ({maha:.1f} > {gate_threshold:.1f})")
                        else:
                            logger.debug(f"Mahalanobis gate: {maha:.1f} > {gate_threshold:.1f}")
                            self.stats['outliers_rejected'] += 1
                            detection = None
                
            if detection is not None:
                if len(self.position_history) > 0:
                    last_pos = self.position_history[-1]
                    movement = np.sqrt((x_center - last_pos[0])**2 + (y_center - last_pos[1])**2)
                    self.movement_history.append(float(movement))
                
                mahalanobis = self.kalman.correct(x_center, y_center)
                self.mahalanobis_history.append(mahalanobis)
                
                self.lost_frames = 0
                self.consecutive_detections += 1
                
                current_strength = min(self.consecutive_detections / 15.0, 1.0)
                base_noise = 3.0 + (7.0 * (1.0 - current_strength))
                erratic_noise_boost = 4.0 if detector_erratic else 0.0
                target_noise = base_noise + erratic_noise_boost
                self.kalman.set_measurement_noise(target_noise)
                
                if is_very_stable and movement < 5.0:
                    self.stability_lock = self.stability_lock_max
                
                if self.adaptive_noise and len(self.position_history) >= 5:
                    recent_positions = list(self.position_history)[-5:]
                    movements = [np.sqrt((recent_positions[i+1][0] - recent_positions[i][0])**2 + 
                                       (recent_positions[i+1][1] - recent_positions[i][1])**2)
                               for i in range(len(recent_positions)-1)]
                    avg_movement = np.mean(movements)
                    
                    if avg_movement < 5.0:
                        self.kalman.Q *= 0.98
                    elif avg_movement > 50.0:
                        self.kalman.Q *= 1.02
                
                vx, vy = self.kalman.get_velocity()
                self.velocity_history.append((vx, vy))
                
                self.is_tracking = True
                self.last_detection = detection
                self.last_bbox = bbox
                self.stats['successful_tracks'] += 1
                
                self.position_history.append((x_center, y_center))
                self.confidence_history.append(confidence)
                
                self._update_hypotheses(x_center, y_center, confidence)
                
                if self.stats['total_updates'] % 100 == 0:
                    logger.debug(f"Tracker stats: updates={self.stats['total_updates']}, "
                                f"tracks={self.stats['successful_tracks']}, "
                                f"predictions={self.stats['predictions_used']}, "
                                f"outliers={self.stats['outliers_rejected']}, "
                                f"erratic={self.stats['erratic_detections']}, "
                                f"strength={current_strength:.2f}")
                
                return (x_center, y_center, True)
        
        self.lost_frames += 1
        self.consecutive_detections = max(0, self.consecutive_detections - 2)
        logger.debug(f"Lost frames: {self.lost_frames}/{self.max_lost_frames}")
        
        if self.lost_frames == 1:
            self.kalman.set_measurement_noise(8.0)
        elif self.lost_frames == 3:
            self.kalman.set_measurement_noise(12.0)
        
        if self.lost_frames <= self.max_lost_frames and self.kalman.initialized:
            pred_x, pred_y = self.kalman.predict()
            
            if detections_list and len(detections_list) > 0:
                search_radius = min(300.0 + (self.lost_frames * 100.0), 800.0)
                best_match = self._find_best_match(pred_x, pred_y, detections_list, search_radius)
                if best_match is not None:
                    logger.info(f"Recovered track with alternative detection (lost_frames={self.lost_frames})")
                    return self.update(best_match)
            
            self.predicted_position = (pred_x, pred_y)
            self.stats['predictions_used'] += 1
            
            self.position_history.append((pred_x, pred_y))
            self.confidence_history.append(0.0)
            
            if not self.kalman.is_stable():
                logger.debug("Kalman filter unstable, increasing uncertainty")
                self.kalman.P *= 1.3
            
            return (pred_x, pred_y, False)
        else:
            self.is_tracking = False
            self.consecutive_detections = 0
            logger.info(f"Track lost after {self.lost_frames} frames")
            return None
    
    def _compute_iou(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        left1, top1 = x1 - w1/2, y1 - h1/2
        right1, bottom1 = x1 + w1/2, y1 + h1/2
        left2, top2 = x2 - w2/2, y2 - h2/2
        right2, bottom2 = x2 + w2/2, y2 + h2/2
        
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)
        
        if inter_right < inter_left or inter_bottom < inter_top:
            return 0.0
        
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _is_outlier(self, x: float, y: float) -> bool:
        if len(self.position_history) < 5:
            return False
        
        recent = np.array(list(self.position_history)[-10:])
        mean_pos = np.mean(recent, axis=0)
        std_pos = np.std(recent, axis=0)
        
        if std_pos[0] < 1e-6 or std_pos[1] < 1e-6:
            return False
        
        z_score_x = abs((x - mean_pos[0]) / std_pos[0])
        z_score_y = abs((y - mean_pos[1]) / std_pos[1])
        
        return z_score_x > 3.0 or z_score_y > 3.0
    
    def _is_detector_erratic(self) -> bool:
        if len(self.movement_history) < 8:
            return False
        
        recent_movements = list(self.movement_history)[-8:]
        variance = np.var(recent_movements)
        mean_movement = np.mean(recent_movements)
        
        if variance > 800.0:
            return True
        
        if mean_movement > 80.0 and variance > 400.0:
            return True
        
        consecutive_large_jumps = sum(1 for m in recent_movements[-5:] if m > 100.0)
        if consecutive_large_jumps >= 3:
            return True
        
        return False
    
    def _calculate_movement_variance(self) -> float:
        if len(self.movement_history) < 3:
            return 0.0
        return float(np.var(list(self.movement_history)[-10:]))
    
    def _find_best_match(self, pred_x: float, pred_y: float, 
                         detections: List[Tuple[float, float, float, float, float]], 
                         search_radius: float = 300.0) -> Optional[Tuple]:
        if not detections:
            return None
        
        best_detection = None
        best_score = float('inf')
        
        relaxed_conf_threshold = max(0.2, self.min_confidence - 0.15)
        
        for det in detections:
            if len(det) == 6:
                x, y, w, h, conf, cls_id = det
                det_tuple = (x, y, w, h, conf)
            else:
                x, y, w, h, conf = det
                det_tuple = det
            
            if conf < relaxed_conf_threshold:
                continue
            
            distance = float(np.sqrt((x - pred_x)**2 + (y - pred_y)**2))
            if distance > search_radius:
                continue
            
            score = distance / max(conf, 1e-3)
            if score < best_score:
                best_score = score
                best_detection = det_tuple
        
        return best_detection
    
    def _update_hypotheses(self, x: float, y: float, confidence: float):
        max_hypotheses = 5
        self.hypotheses.append((x, y))
        self.hypothesis_scores.append(confidence)
        
        if len(self.hypotheses) > max_hypotheses:
            self.hypotheses.pop(0)
            self.hypothesis_scores.pop(0)
    
    def get_velocity(self) -> Tuple[float, float]:
        if self.kalman.initialized:
            return self.kalman.get_velocity()
        
        if len(self.position_history) < 2:
            return 0.0, 0.0
        
        recent_positions = list(self.position_history)[-5:]
        vx = (recent_positions[-1][0] - recent_positions[0][0]) / len(recent_positions)
        vy = (recent_positions[-1][1] - recent_positions[0][1]) / len(recent_positions)
        
        return vx, vy
    
    def get_acceleration(self) -> Tuple[float, float]:
        if self.kalman.initialized:
            return self.kalman.get_acceleration()
        return 0.0, 0.0
    
    def get_avg_confidence(self, window: int = 10) -> float:
        if len(self.confidence_history) == 0:
            return 0.0
        recent = list(self.confidence_history)[-window:]
        return sum(recent) / len(recent)
    
    def reset(self):
        logger.info("Resetting BallTracker")
        dt = self.kalman.dt
        self.kalman = ExtendedKalmanFilter(dt=dt, process_noise=0.01, measurement_noise=5.0)
        self.lost_frames = 0
        self.consecutive_detections = 0
        self.is_tracking = False
        self.position_history.clear()
        self.confidence_history.clear()
        self.velocity_history.clear()
        self.mahalanobis_history.clear()
        self.movement_history.clear()
        self.last_detection = None
        self.predicted_position = None
        self.last_bbox = None
        self.hypotheses.clear()
        self.hypothesis_scores.clear()
        self.stability_lock = 0
        self.stats = {
            'total_updates': 0,
            'successful_tracks': 0,
            'predictions_used': 0,
            'noise_adaptations': 0,
            'outliers_rejected': 0,
            'erratic_detections': 0
        }
    
    def get_state(self) -> Dict:
        velocity = self.get_velocity()
        acceleration = self.get_acceleration()
        
        return {
            'is_tracking': self.is_tracking,
            'lost_frames': self.lost_frames,
            'last_position': self.position_history[-1] if self.position_history else None,
            'avg_confidence': self.get_avg_confidence(),
            'velocity': velocity,
            'velocity_magnitude': np.sqrt(velocity[0]**2 + velocity[1]**2),
            'acceleration': acceleration,
            'kalman_stable': self.kalman.is_stable(),
            'track_id': self.track_id,
            'stats': self.stats.copy()
        }
    
    def get_stats(self) -> Dict:
        return self.stats.copy()
