import numpy as np
from ligotools import readligo

TEST_FILE = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"

def test_loaddata_returns_arrays_and_dict():
    """Check that loaddata returns strain and time arrays, and a channel dictionary"""
    strain, time, channel_dict = readligo.loaddata(TEST_FILE)
    
    assert isinstance(strain, np.ndarray), "Strain should be a numpy array"
    assert isinstance(time, np.ndarray), "Time should be a numpy array"
    assert isinstance(channel_dict, dict), "Channel dict should be a dictionary"
    
    assert len(strain) == len(time), "Strain and time arrays should have same length"

def test_channel_dict_contains_default():
    """Check that channel dictionary contains DEFAULT key"""
    _, _, channel_dict = readligo.loaddata(TEST_FILE)
    
    assert 'DEFAULT' in channel_dict, "Channel dictionary should contain 'DEFAULT' key"
    assert isinstance(channel_dict['DEFAULT'], np.ndarray), "'DEFAULT' channel should be an array"
