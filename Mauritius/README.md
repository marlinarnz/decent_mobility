# How-to

## Install the Python environment

Install the `quetzal-lite` environment as described here: https://github.com/marlinarnz/quetzal-lite

## Run the model

All network files are included in the `model` folder, ready to use. Just the origin-destination tables must be generated for each region with the correspondingly named notebook in the `notebooks` folder: Load the network files from the `model` folder (or generate them anew by fetching OSM data) and run the pathfinders in the bottom section of the notebook.

## Decent mobility analysis

With the origin-destination tables generated, you are able to run the simplified activity-based transport model. Open the `decent_mobility_modelling` notebook, enter the desired region name ("Port Louis" or "Vacoas") in the second cell, run it, have a look at the figures, and change it to your needs.

All output plots and tables are saved in the `output` folder.
