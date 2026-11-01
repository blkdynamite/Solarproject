"""Demand-response M&V for fleets of distributed energy resources.

Simulates a smart-thermostat fleet with known ground truth, builds a SQL data
layer, and compares baseline, ML and randomised-control-group estimators of
delivered load reduction, with bootstrap uncertainty and dollar impact.
"""

__version__ = "1.0.0"
