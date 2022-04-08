from src.zoo.pre_processing import *
import pandas as pd

link = "https://raw.githubusercontent.com/poddarswakhar/dump/main/zoo.data"
header = ["animalName", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
          "predator", "toothed", "backbone", "breathes", "venomous", "fins",
          "legs", "tail", "domestic", "catsize", "type"]

# expected df
expected = pd.read_csv("https://raw.githubusercontent.com/poddarswakhar/dump/main/exp.csv")


def test_pre_process():
    actual = pre_process(link, header)

    # checking the colm and row
    act_shape = actual.shape
    exp_shape = expected.shape

    if act_shape == exp_shape:
        assert True
    else:
        assert False

    # checking the content of actual and expected
    if actual.equals(expected):
        assert True
    else:
        assert False
