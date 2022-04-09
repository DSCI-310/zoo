# zoo

Package created for DSCI310-2021W2-Group7 zoo analysis including five useful functions: [line_plot](src/zoo/line_plot.py), [final_Model](src/zoo/train_and_predict_model.py), [std_acc](src/zoo/std_acc.py), 
[pre_processing](src/zoo/pre_processing.py) and [para_optimize](src/zoo/para_optimize.py).

## Installation

```bash
pip install zoo
```

## Usage

1. `line_plot` is a function which plots a linear relationship.

   Example: `line_plot(Ks, mean, std, "x-axis", "y-axis", "population distribution")`

2. `final_Model` is a function which creates the final model of the specific algorithm.

   Example:  `final_Model(algorithm, threshold, X_train, X_test, y_train, y_test, X, y)`

3. `std_acc` is a function which generates the standard deviation of predicted array and observed array associated with each k value.

   Example: ` std_Acc(yhat, y_test, 10)`

4. `pre_process` is a function which downloads the data from the link without header and add the desired header to the data.

   Example: `pre_process(link, header)`

5. `para_optimize` is a function which optimizes hyper-parameters for a model.

   Example: `para_optimize(knn, param_grid, 5)`

Detail information see file [examples](docs/example.ipynb).


## Contributing

Interested in contributing? Check out the [contributing guidelines](CONTRIBUTING.md). Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.

## License

`zoo` was created by DSCI310-2021W2-Group7. It is licensed under the terms of the MIT license, see [liscense](LICENSE.md).

## Credits

`zoo` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
