import pytest
import matplotlib
import numpy as np
from src.zoo.line_plot import *

Ks = 10
k = 0
mean = np.zeros((Ks-1))
std = np.zeros((Ks-1))
x = "x-axis"
y = "y-axis"
name = "name of the plot"

def test_line_plot_type():
    assert isinstance(line_plot(Ks, mean, std, x, y, name), matplotlib.figure.Figure)
    
def test_line_plot_input_type():
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert isinstance(name, str)
    assert isinstance(Ks, int)
    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)
    
def test_line_plot_wrong_k():
    assert line_plot(k, mean, std, x, y, name), "wrong value for K"
    
def test_line_plot_wrong_input():
    assert line_plot(k, "a", std, x, y, name), "wrong type of input"

@pytest.mark.mpl_image_compare(baseline_dir='baseline',
                               filename='test_line_plot.png')
def test_line_plot_figure():
    return line_plot(Ks, mean, std, x, y, name)

    