from __future__ import annotations

from cobras.cobras_core.evaluation_ops import EvaluationOpsMixin
from cobras.cobras_core.population_ops import PopulationOpsMixin
from cobras.cobras_core.skill_ops import SkillOpsMixin


class CobrasOperatorsMixin(EvaluationOpsMixin, PopulationOpsMixin, SkillOpsMixin):
    """Compatibility facade for all task and population operations."""

