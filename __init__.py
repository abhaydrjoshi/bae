"""BAE package — convenience exports for the consolidated engine.

This package exports the `engine` submodule and commonly used classes
so `import bae` provides direct access to core functionality.
"""

__version__ = "0.1.0"

from . import engine as engine
from .engine import (
	main_demo,
	Trainer,
	TriadicMesh,
	LatencyField,
	SynapticDynamics,
	ContextModulator,
	DorsicLayer,
	IonicLayer,
	CorinthianLayer,
	ExperimentSuite,
	MetricsLogger,
	FullMCTS,
)

__all__ = [
	"engine",
	"main_demo",
	"Trainer",
	"TriadicMesh",
	"LatencyField",
	"SynapticDynamics",
	"ContextModulator",
	"DorsicLayer",
	"IonicLayer",
	"CorinthianLayer",
	"ExperimentSuite",
	"MetricsLogger",
	"FullMCTS",
	"__version__",
]
