"""ElectroPolyBench: retrospective active-learning benchmark for electropolymerization / OMIEC materials.

v1 scope: a dataset- and strategy-agnostic retrospective pool-based AL replay harness,
proven end-to-end on the real PANDA electrodeposited PEDOT:PSS dataset (with a documented
synthetic-surface fallback). See README for the research question and v2 roadmap.
"""

from . import acquisitions, metrics, replay, surrogates

__all__ = ["surrogates", "acquisitions", "replay", "metrics"]
__version__ = "0.1.0"
