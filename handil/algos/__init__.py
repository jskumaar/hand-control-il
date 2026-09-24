"""Algorithm registry.

Adding one: subclass `Policy`, decorate with `@register("name")`, import it
here. Diffusion Policy is the obvious next addition -- it shares ACT's chunked
SequenceDataset path, so only the module is needed.
"""
from .base import Policy, available, build, register  # noqa: F401
from . import bc, act, ibc  # noqa: F401  (import for registration side effect)

__all__ = ["Policy", "build", "register", "available"]
