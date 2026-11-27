"""Location of the bee dataset on disk.

The varroa experiments resolve their default dataset folders under
``DATA_ROOT``. Set the ``BEES_DATA_ROOT`` environment variable to point it at
your copy of the dataset; by default it is ``./data/DatasetApi_Ceschi``
relative to the current working directory.
"""
import os

DATA_ROOT = os.environ.get("BEES_DATA_ROOT", os.path.join("data", "DatasetApi_Ceschi"))
