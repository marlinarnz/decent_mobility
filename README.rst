Decent mobility
******************************

An EDITS 2024 fast-track project.

Contents:

- ``mobility/``: A Python module.

  - ``retreat.py``: Code from 2012-12-09 writing retreat.
    Use ``python mobility/retreat.py`` for a demonstration.

  - ``persona.py`` and ``alternative.py``: Code that simulates mobility choices of a ``persona`` that has several ``alternative`` for satisfying their "decent" mobility needs.
    Use ``python mobility/test_persona.py`` for a demonstration.

  - ``scratch/``: temporary and working files.

    - ``trips.py``, ``trips_xarray.py``: example trip-based calculation of travel distance, time, and money.
      Usage::

        $ pip install requirements.txt
        $ python trips.py

- ``Mauritius``: Contains a simplified activity-based model for two examplary regions on Mauritius. Contains a description on how to run the model.

© 2024–2025 `contributors <https://github.com/marlinarnz/decent_mobility_infrastructure/graphs/contributors>`_.
Code licensed under GNU GPL v3.0 unless otherwise specified.
