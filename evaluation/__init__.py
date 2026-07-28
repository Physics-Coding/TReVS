"""Import-safe evaluation utilities for the reproducibility package.

Public functions are exported by their concrete modules, for example
``evaluation.aggregate_metrics`` and ``evaluation.mme.calculation``. Keeping
the package initializer empty avoids preloading CLI modules during ``-m`` use.
"""
