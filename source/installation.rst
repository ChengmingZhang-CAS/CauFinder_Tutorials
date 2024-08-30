Installation
============

This guide provides a quick and straightforward way to install CauFinder.

Clone the Repository
--------------------

First, clone the CauFinder repository from GitHub:

.. code-block:: bash

   git clone https://github.com/ChengmingZhang-CAS/CauFinder-main.git

Set Up Conda Environment
------------------------

Create and activate a new Conda environment using the provided `environment.yml` file:

.. code-block:: bash

   conda env create -n caufinder_env -f environment.yml
   conda activate caufinder_env

Install Additional Dependencies
-------------------------------

If you need additional dependencies, install them using `pip`:

.. code-block:: bash

   pip install -r requirements.txt

Optional: Install GUROBI
------------------------

For advanced features like master regulator identification, install GUROBI:

.. code-block:: bash

   pip install gurobipy

GUROBI requires a license to run. Visit the [GUROBI](https://www.gurobi.com/) website for academic and trial licenses.

If GUROBI is not available, you can use [SCIP](https://www.scipopt.org/) as an alternative, though results may vary.


