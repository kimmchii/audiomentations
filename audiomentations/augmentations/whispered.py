import pyworld as pw
from scipy.signal.windows import triang
from scipy.signal import lfilter
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform

class Whispered(BaseWaveformTransform):
    """Convert the audio to a whispered version"""
    supports_multichannel = False

    def __init__(self, window_width: int=400, step_coeff_update_size: float=0.01, order: int=5, p: float=0.5):
        """
        :param window_width: The width of the triangular window in Hz. (The suggested value is 400 Hz.)
        :param step_coeff_update_size: The step size for updating the coefficients of the adaptive inverse filter. (The suggested value is 0.01.)
        :param order: The order of the adaptive inverse filter. (The suggested value is 5.)
        :param p: The probability of applying this transform.
        """
        super().__init__(p)
        assert window_width is not None
        self.window_width = window_width
        self.step_coeff_update_size = step_coeff_update_size
        self.order = order

    def iterative_adaptive_inverse_filter(self, samples: NDArray[np.float32]):
        # Apply adaptive inverse filter to the audio.

        # Initialize the filter coefficients.
        filter_coefficients = np.zeros(self.order)
        # Initialize the residual signal.
        residual_signal = np.zeros_like(samples)

        # Iterate over the audio samples.
        for i in range(len(samples)):
            sample_vector = np.asarray([samples[i-j] if i -j >= 0 else 0 for j in range(self.order)])

            # Compute the output of the adaptive inverse filter.
            output_signal = np.dot(filter_coefficients, sample_vector)

            # Compute the error (the difference between the original and the estimated output signal).
            error = samples[i] - output_signal

            # Update the filter coefficients.
            filter_coefficients += self.step_coeff_update_size * error * sample_vector

            # Update the residual signal.
            residual_signal[i] = error

        return residual_signal

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        samples = samples.astype(np.float64)

        # Apply the iterative adaptive inverse filter to the audio.
        samples = self.iterative_adaptive_inverse_filter(samples)

        f0, spectral_envelope, aperiodic_spectral_envelope = pw.wav2world(samples, sample_rate)

        # Modify f0 to 0.
        f0 = np.zeros_like(f0)

        # Apply a triangular window to the spectrogram.
        window_size = int((self.window_width / sample_rate) * spectral_envelope.shape[0])
        window = triang(window_size)
        window = window / np.sum(window)
        smooth_spectral_envelope = np.zeros_like(spectral_envelope)

        for i in range(spectral_envelope.shape[0]):
            smooth_spectral_envelope[i, :] = lfilter(window, 1, spectral_envelope[i, :])

        # Synthesize the whispered audio.
        samples = pw.synthesize(f0, smooth_spectral_envelope, aperiodic_spectral_envelope, sample_rate)
        return samples.astype(np.float32)

        

        
        

        
