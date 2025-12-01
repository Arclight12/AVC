import numpy as np
import scipy.signal

class UniversalPreprocessor:
    def __init__(self, target_rate=1000, fixed_length=None, apply_smoothing=False):
        """
        Universal Preprocessor for Biosignals.
        
        Args:
            target_rate (int): Target sampling rate in Hz.
            fixed_length (int, optional): Fixed length to pad/trim to.
            apply_smoothing (bool): Whether to apply simple smoothing.
        """
        self.target_rate = target_rate
        self.fixed_length = fixed_length
        self.apply_smoothing = apply_smoothing

    def process(self, signal, original_rate):
        """
        Process a raw biosignal.
        
        Args:
            signal (np.ndarray): Shape (Time, Channels) or (Channels, Time).
            original_rate (int): Original sampling rate of the signal.
            
        Returns:
            np.ndarray: Processed signal (Time, Channels).
        """
        # Ensure (Time, Channels)
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
            
        # 1. Resample
        if original_rate != self.target_rate:
            num_samples = int(len(signal) * self.target_rate / original_rate)
            signal = scipy.signal.resample(signal, num_samples)
            
        # 2. Z-score Normalization (per channel)
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0) + 1e-8
        signal = (signal - mean) / std
        
        # 3. Smoothing (Optional)
        if self.apply_smoothing:
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            for i in range(signal.shape[1]):
                signal[:, i] = np.convolve(signal[:, i], kernel, mode='same')

        # 4. Padding/Trimming
        if self.fixed_length:
            if len(signal) > self.fixed_length:
                signal = signal[:self.fixed_length]
            else:
                pad_len = self.fixed_length - len(signal)
                if pad_len > 0:
                    signal = np.pad(signal, ((0, pad_len), (0, 0)), mode='constant')
                
        return signal.astype(np.float32)
