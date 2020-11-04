Constraint-based Learning with Neural Networks
==================================================

Notebooks
----------
This repository contains two notebooks which will guide you step-by-step towards
the implementation of learning of and with constraints in Pytorch.

- `Learning with constraints <https://github.com/pietrobarbiero/constraint-learning/blob/master/notebooks/learning_with_constraints_digits.ipynb>`_:
  learn how to train a NN with human-driven constraints

* - .. figure:: https://github.com/pietrobarbiero/constraint-learning/blob/master/img/learning_with_constraints.png
        :height: 200px

- `Learning of constraints <https://github.com/pietrobarbiero/constraint-learning/blob/master/notebooks/learning_of_constraints_digits.ipynb>`_:
  learn how to make a NN learn how to explain its predictions with logic

* - .. figure:: https://github.com/pietrobarbiero/constraint-learning/blob/master/img/learning_of_constraints.png
        :height: 200px

Theory
--------
Theoretical foundations can be found in the following papers.

Learning of constraints::

    @inproceedings{ciravegna2020constraint,
      title={A Constraint-Based Approach to Learning and Explanation.},
      author={Ciravegna, Gabriele and Giannini, Francesco and Melacci, Stefano and Maggini, Marco and Gori, Marco},
      booktitle={AAAI},
      pages={3658--3665},
      year={2020}
    }

Learning with constraints::

    @inproceedings{marra2019lyrics,
      title={LYRICS: A General Interface Layer to Integrate Logic Inference and Deep Learning},
      author={Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and Gori, Marco},
      booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
      pages={283--298},
      year={2019},
      organization={Springer}
    }

Constraints theory in machine learning::

    @book{gori2017machine,
      title={Machine Learning: A constraint-based approach},
      author={Gori, Marco},
      year={2017},
      publisher={Morgan Kaufmann}
    }


Authors
-------

`Pietro Barbiero <http://www.pietrobarbiero.eu/>`__

Licence
-------

Copyright 2020 Pietro Barbiero.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.