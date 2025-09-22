pomdp_py
========

Source Framework
----------------
This repository was built on the POMDP-PY framework:
`pomdp_py <https://h2r.github.io/pomdp-py>`_.
More information about the framework can be found in the following sources:

- `Full documentation <https://h2r.github.io/pomdp-py>`_
- `Installation instructions <https://h2r.github.io/pomdp-py/html/installation.html>`_

Citation
~~~~~~~~
If you use this framework, please cite:

.. code-block:: bibtex

   @inproceedings{zheng2020pomdp_py,
     title     = {pomdp\_py: A Framework to Build and Solve POMDP Problems},
     author    = {Zheng, Kaiyu and Tellex, Stefanie},
     booktitle = {ICAPS 2020 Workshop on Planning and Robotics (PlanRob)},
     year      = {2020},
     url       = {https://icaps20subpages.icaps-conference.org/wp-content/uploads/2020/10/14-PlanRob_2020_paper_3.pdf},
     note      = {Arxiv link: \url{https://arxiv.org/pdf/2004.10099.pdf}}
   }

Quick Run Instructions
----------------------
1. Set up a virtual environment. E.g.:

  .. code-block:: bash

     python3 -m venv venv

2. Install main package and dependencies by navigating to the ``pomdp-py-porpp`` folder (where ``setup.py`` is located) and running:

   .. code-block:: bash

      pip install -e .

3. Scripts can be run by navigating to ``pomdp-py-porpp/scripts``.
   For example, a parallelized run using GNU Parallel might be executed as follows:

   .. code-block:: bash

      nohup parallel -j 8 -a script_name.sh
