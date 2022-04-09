from src.pre_processing import *
import pandas as pd

# test two expected valid cases
link = "https://raw.githubusercontent.com/poddarswakhar/dump/main/zoo.data"
header = ["animalName", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
          "predator", "toothed", "backbone", "breathes", "venomous", "fins",
          "legs", "tail", "domestic", "catsize", "type"]

link2 = "https://raw.githubusercontent.com/jossiej00/iris_data/main/iris_raw.csv"
header2 = ["Sepal.Length", "Sepal.Width", "Petal.Width", "Petal.Length", "Species"]
# expected df
expected = pd.read_csv("https://raw.githubusercontent.com/poddarswakhar/dump/main/exp.csv")
expected2 = pd.read_csv("https://raw.githubusercontent.com/jossiej00/iris_data/main/iris.csv")

## test one invalid input case
invalid_header = 2
def test_pre_process():
    ## test two expected valid cases
    actual = pre_process(link, header)
    actual2 = pre_process(link2,header2)

    # checking the colm and row
    act_shape = actual.shape
    exp_shape = expected.shape

    act_shape2 = actual2.shape
    exp_shape2 = expected2.shape

    if act_shape == exp_shape:
        assert True
    else:
        assert False

    if act_shape2 == exp_shape2:
        assert True
    else:
        assert False

    # checking the content of actual and expected
    if actual.equals(expected):
        assert True
    else:
        assert False

    if actual2.equals(expected2):
        assert True
    else:
        assert False

    ## test if warning msg pops up
    msg = pre_process(link, invalid_header)
    if msg == "Input 'link' should be a string type web link and input 'header' should be a list!":
        assert  True
    else:
        assert False
