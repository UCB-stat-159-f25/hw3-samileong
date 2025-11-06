import os
import numpy as np
from ligotools import readligo

TEST_FILE = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"

def test_loaddata_output_structure():
    """Check that loaddata returns strain, time, and metadata"""
    strain, time, meta = readligo.loaddata(TEST_FILE)
    
    assert isinstance(strain, np.ndarray), "Strain should be a numpy array"
    assert isinstance(time, np.ndarray), "Time should be a numpy array"
    assert isinstance(meta, dict), "Metadata should be a dictionary"
    
    assert len(strain) == len(time), "Strain and time arrays should have the same length"

def test_metadata_contains_expected_keys():
    """Check that metadata has standard keys from LOSC files"""
    _, _, meta = readligo.loaddata(TEST_FILE)
    for key in ["Detector", "UTCstart", "Duration"]:
        assert key in meta, f"Metadata should contain '{key}' key"
