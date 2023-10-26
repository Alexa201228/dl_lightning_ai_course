:orphan:

.. _install:


############
Installation
############

We strongly recommend to create a virtual environment first.
Don't know what this is? Follow our `beginner guide here <install_beginner.rst>`_.

**Requirements**

* Python 3.8.x or later (3.8.x, 3.9.x, 3.10.x)

Or read the `Apple Silicon Macs installation article <installation_mac.rst>`_ or the `Windows installation article <installation_win.rst>`_.

----

****************
Install with pip
****************

0.  Activate your virtual environment.

You can use `conda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_ or `pyenv <https://github.com/pyenv/pyenv>`_.

1.  Install the ``lightning-hpo`` package

.. code:: bash

   git clone https://github.com/Lightning-AI/lightning-hpo && cd lightning-hpo

   pip install -e . -r requirements.txt --find-links https://download.pytorch.org/whl/cpu/torch_stable.html

.. note:: This is only temporary as some changes are being released to PiPy


2. Makes sure everything works locally

.. code-block:: bash

   python -m lightning run app app.py

3. Makes sure everything works in the cloud

.. code-block:: bash

   PACKAGE_LIGHTNING=1 python -m lightning run app app.py --cloud

----

**********
Next Steps
**********

.. raw:: html

   <br />
   <div class="display-card-container">
      <div class="row">

.. displayitem::
   :header: Run a Sweep with PyTorch Lightning
   :description: Discover PyTorch Lightning and train your first Model.
   :col_css: col-md-6
   :button_link: workflows/optimize_with_pytorch_lightning.html
   :height: 180

.. displayitem::
   :header: The Training Studio App
   :description: Manage Sweeps and Tools to accelerate Training.
   :col_css: col-md-6
   :button_link: training_studio.html
   :height: 180

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

.. raw:: html

   <div style="display:none">

----

***************
Troubleshooting
***************

.. note:: On MacOS, if you face an OSError: Too many open files, you can increase your Mac process limit with: **ulimit -Sn 50000**
