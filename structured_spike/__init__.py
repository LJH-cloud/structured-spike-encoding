"""Structured spike encoding and controlled temporal initialization."""
from .encoding import StructuredSpikeEncoder
from .metrics import metrics_per_sample, aggregate_capture_macro

__version__ = "0.1.0"
__all__ = ["StructuredSpikeEncoder", "metrics_per_sample", "aggregate_capture_macro"]
