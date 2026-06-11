Decent mobility
******************************

An EDITS 2024 fast-track project.

Contents:

- ``Seville``: An updated, improved version of the disaggregated modelling in ``Mauritius`` for the city of Seville and its functional urban area (FUA). Run the network generation notebooks and generate results in the ``decent_mobility_modelling_dynamic`` notebook.

- ``Mauritius``: Contains a simplified activity-based model for two examplary regions on Mauritius, as showcased in the [corresponding scientific paper](https://doi.org/10.1016/j.erss.2025.104306). Contains a description on how to run the model.

- ``mobility/``: An experimental Python module.

  - ``retreat.py``: Code from 2024-12-09 writing retreat.
    Use ``python mobility/retreat.py`` for a demonstration.

  - ``persona.py`` and ``alternative.py``: Code that simulates mobility choices of a ``persona`` that has several ``alternative`` for satisfying their "decent" mobility needs.
    Use ``python mobility/test_persona.py`` for a demonstration.

  - ``scratch/``: temporary and working files.

    - ``trips.py``, ``trips_xarray.py``: example trip-based calculation of travel distance, time, and money.
      Usage::

        $ pip install requirements.txt
        $ python trips.py


© 2024–2026 `contributors <https://github.com/marlinarnz/decent_mobility/graphs/contributors>`_.
Code licensed under GNU GPL v3.0 unless otherwise specified.
