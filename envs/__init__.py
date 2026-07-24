"""
Export package init file.

When you write `from export import PolicyExporter`, Python looks here first.
This package handles converting the trained Python brain into formats that real physical robots can run.
"""

from export.exporter import PolicyExporter  # Converts PyTorch model -> ONNX file + C++ microcontroller header

# __all__ lists the public classes that can be imported from this package
__all__ = ["PolicyExporter"]
