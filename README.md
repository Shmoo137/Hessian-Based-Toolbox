# Hessian-based toolbox for more reliable and interpretable machines learning physics

[![DOI](https://zenodo.org/badge/DOI/?.svg)](https://doi.org/?)

## Hessian-based toolbox on the example of the CNN trained to recognize between the Luttinger liquid (LL) and the charge density wave I (CDW-I)
The aim of this code is to give a possibility of applying a Hessian-based toolbox to your own ML model and data. We described this toolbox in detail in the paper ["Hessian-based toolbox for more reliable and interpretable machines learning physics"](https://arxiv.org/abs/?) by A. Dawid, P. Huembeli, M. Tomza, M. Lewenstein, and A. Dauphin. The code contains:
- Jupyter notebook `Hessian-based_notebook.ipynb`
- `utility_general.py` and `utility_plots.py` with utility and plotting functions,
- `architecture.py` specyfing the model we used, add your own,
- `data_loader.py` to load data sets from folder `data`,
- `influence_functions.py` containing functions to compute the [influence functions](http://proceedings.mlr.press/v70/koh17a.html) and [RelatIFs](https://proceedings.mlr.press/v108/barshan20a),
- `rue.py` containing functions to compute the [Resampling Uncertainty Estimation (RUE)](https://proceedings.mlr.press/v89/schulam19a),
- `lees.py` containing functions to compute the [Extrapolation Score with Local Ensembles (LEs)](https://openreview.net/forum?id=BJl6bANtwH),
- folder `data` containing the original ground states with labels being the phases LL (0) or CDW (1),
- folder `model` containing the original model we used and the mask used to shuffle the training data in a way possible to follow, as well as the computed Hessian of the training loss at the minimum
- folder `toolbox_output` with the final results for our exemplary model and data.

All data contained in folder `toolbox_output` can be reproduced in the `Hessian-based_notebook.ipynb` notebook.
Data contained in `model` and `data` can be reproduced using [this notebook](https://doi.org/10.5281/zenodo.3759432).

Code was written by Anna Dawid (University of Warsaw & ICFO) with help of Alexandre Dauphin (ICFO) and Patrick Huembeli (EPFL)