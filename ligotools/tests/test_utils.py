import numpy as np
from scipy.io import wavfile
import os
from ligotools.utils import whiten, write_wavfile, reqshift, plot_matched_filter_psd

def test_whiten():
    # Create a simple sine wave
    dt = 1/4096
    t = np.arange(0, 1, dt)
    strain = np.sin(2*np.pi*50*t)  # 50 Hz sine
    psd = lambda f: np.ones_like(f)  # flat PSD
    whitened = whiten(strain, psd, dt)
    
    # The whitened output should have same length as input
    assert len(whitened) == len(strain)
    # Values should not be exactly equal to the input (should be transformed)
    assert not np.allclose(whitened, strain)

def test_write_wavfile(tmp_path):
    # Create a temporary file
    filename = tmp_path / "test.wav"
    fs = 1024
    data = np.sin(2*np.pi*10*np.arange(0,1,1/fs))
    
    # Call write_wavfile
    write_wavfile(filename, fs, data)
    
    # Check the file exists
    assert os.path.exists(filename)
    
    # Check that reading it back returns expected shape
    rate, read_data = wavfile.read(filename)
    assert rate == fs
    assert len(read_data) == len(data)
