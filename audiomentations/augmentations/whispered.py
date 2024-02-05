import pyworld as pw
from scipy.signal import triang, lfilter
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform

class Whispered(BaseWaveformTransform):
    """Convert the audio to a whispered version"""
    supports_multichannel = False

    def __init__(self, window_width:int=400, p: float=0.5):
        """
        :param window_width: The width of the triangular window in Hz. (The suggested value is 400 Hz.)
        :param p: The probability of applying this transform.
        """
        super().__init__(p)
        assert window_width is not None
        self.window_width = window_width

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        samples = samples.astype(np.float64)
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

        

        
        

        