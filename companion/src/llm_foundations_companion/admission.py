"""Framework-free admission dispatch for resolved S2 and later operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .scheduler import AdmissionPlanner as LegacyAdmissionPlanner
from .training_store import AdmissionPlan, TrainingStore


S2_OPERATIONS = frozenset(
    {
        "tokenizer_train",
        "tiny_train",
        "tiny_resume",
        "evaluate",
        "context_preview",
        "generate",
    }
)


class AdmissionPlanner:
    """Resolve S2 registry identities while retaining bounded later-slice plans."""

    def __init__(
        self,
        training_store: TrainingStore,
        *,
        fallback: Any | None = None,
    ) -> None:
        self.training_store = training_store
        self.fallback = fallback or LegacyAdmissionPlanner()

    def resolve(self, request: Mapping[str, Any]) -> AdmissionPlan | None:
        if request.get("operation") in S2_OPERATIONS:
            return self.training_store.resolve(request)
        return None

    def plan(self, request: Mapping[str, Any]) -> dict[str, int]:
        return dict(self.fallback.plan(request))


__all__ = ["AdmissionPlanner", "S2_OPERATIONS"]
