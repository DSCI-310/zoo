import pytest
import matplotlib
import numpy as np
from src.zoo.line_plot import *

Ks = 10
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
    
@pytest.mark.mpl_image_compare(baseline_dir='baseline',
                               filename='test_line_plot.png')
def test_line_plot_figure():
    return line_plot(Ks, mean, std, x, y, name)

    