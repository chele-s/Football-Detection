import sys
import os
import time
import random
import math

# Add app to path
sys.path.append(os.getcwd())

try:
    import numpy as np
except ImportError:
    print("Numpy not found, using pure python fallback for test")
    np = None

from app.camera.one_euro_filter import OneEuroFilter

def test_one_euro_filter():
    print("Testing One Euro Filter...")
    
    # Generate synthetic signal: Sine wave + Noise
    duration = 5.0
    fps = 30.0
    steps = int(duration * fps)
    t = [i/fps for i in range(steps)]
    
    # Signal: Sine wave
    signal = [100 * math.sin(2 * math.pi * 0.5 * ti) + 500 for ti in t]
    
    # Noise: Random
    noise = [random.gauss(0, 5.0) for _ in range(steps)]
    noisy_signal = [s + n for s, n in zip(signal, noise)]
    
    # Filter
    f = OneEuroFilter(freq=fps, min_cutoff=0.1, beta=0.05, d_cutoff=1.0)
    
    filtered_signal = []
    start_time = time.time()
    
    for i in range(steps):
        val = noisy_signal[i]
        timestamp = t[i]
        
        # Filter (x, y) - we just use x here
        res_x, _ = f(val, 0.0, timestamp=timestamp)
        filtered_signal.append(res_x)
        
    end_time = time.time()
    print(f"Processed {steps} frames in {(end_time - start_time)*1000:.2f}ms")
    
    # Calculate metrics
    if np:
        signal_np = np.array(signal)
        filtered_np = np.array(filtered_signal)
        noisy_np = np.array(noisy_signal)
        
        rmse_noise = np.sqrt(np.mean((noisy_np - signal_np)**2))
        rmse_filtered = np.sqrt(np.mean((filtered_np - signal_np)**2))
        
        diff_noise = np.sum(np.abs(np.diff(noisy_np)))
        diff_filtered = np.sum(np.abs(np.diff(filtered_np)))
    else:
        # Pure python metrics
        mse_noise = sum((n - s)**2 for n, s in zip(noisy_signal, signal)) / steps
        rmse_noise = math.sqrt(mse_noise)
        
        mse_filtered = sum((f - s)**2 for f, s in zip(filtered_signal, signal)) / steps
        rmse_filtered = math.sqrt(mse_filtered)
        
        diff_noise = sum(abs(noisy_signal[i+1] - noisy_signal[i]) for i in range(steps-1))
        diff_filtered = sum(abs(filtered_signal[i+1] - filtered_signal[i]) for i in range(steps-1))

    print(f"RMSE Noise: {rmse_noise:.2f}")
    print(f"RMSE Filtered: {rmse_filtered:.2f}")
    print(f"Noise Reduction: {(1 - rmse_filtered/rmse_noise)*100:.1f}%")
    
    print(f"Jitter (Noise): {diff_noise:.2f}")
    print(f"Jitter (Filtered): {diff_filtered:.2f}")
    print(f"Jitter Reduction: {(1 - diff_filtered/diff_noise)*100:.1f}%")

    if rmse_filtered < rmse_noise and diff_filtered < diff_noise:
        print("✅ TEST PASSED: Filter reduced noise and jitter.")
    else:
        print("❌ TEST FAILED: Filter did not improve signal.")

if __name__ == "__main__":
    test_one_euro_filter()
