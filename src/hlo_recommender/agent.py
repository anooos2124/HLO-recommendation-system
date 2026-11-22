"""Hybrid Knowledge-Grounded Recommender Agent for HLO."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from .knowledge import HLO, LEVELS
from .models.bkt import BKTParameters, LearnerStore, load_parameters
from .pipeline import Runtime
from .recommendation import Resource


@dataclass(frozen=True)
class AgentToolResult:
    name: str
    status: str
    data: Any
    message: str = ""


class OntologyGroundingTool:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def execute(self, question: str) -> AgentToolResult:
        skills = self.runtime.grounder.skills(question)
        if not skills:
            return AgentToolResult(
                name="grounding",
                status="clarification_needed",
                data=[],
                message="No matching clinical skills identified in the HLO ontology for this query."
            )
        details = []
        for s in skills:
            label = s.split("#")[-1] if "#" in s else s.split("/")[-1]
            details.append({"uri": s, "label": label})
        return AgentToolResult(
            name="grounding",
            status="success",
            data=details,
            message=f"Successfully grounded question to {len(details)} clinical skill(s)."
        )


class KnowledgeTracingTool:
    def __init__(self, store: Optional[LearnerStore], params: BKTParameters):
        self.store = store
        self.params = params

    def execute(self, learner_id: str, skills: List[str]) -> AgentToolResult:
        state = self.store.snapshot(learner_id) if self.store else {}
        mastery_map = {}
        for s in skills:
            raw = state.get(s, self.params.initial)
            m = raw["mastery"] if isinstance(raw, dict) else float(raw)
            mastery_map[s] = {
                "mastery": round(m, 4),
                "gap": round(1.0 - m, 4),
                "level": "Proficient" if m >= 0.8 else ("Developing" if m >= 0.4 else "Novice")
            }
        return AgentToolResult(
            name="knowledge_tracing",
            status="success",
            data=mastery_map,
            message=f"Retrieved BKT mastery state for learner '{learner_id}'."
        )


class RecommendationRankingTool:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def execute(self, question: str, context: dict, state: Optional[dict] = None) -> AgentToolResult:
        generator = self.runtime.generator("template")
        rec = self.runtime.recommend(question, context, generator, state=state, adaptive=True)
        return AgentToolResult(
            name="recommendation_ranking",
            status=rec.get("status", "ok"),
            data=rec,
            message=f"Retrieved and ranked {len(rec.get('ranked', []))} resources."
        )


class HLORecommenderAgent:
    """Agent orchestrating ontology grounding, BKT knowledge state, and adaptive ranking."""

    def __init__(self, runtime: Runtime, bkt_params: Optional[BKTParameters] = None, store: Optional[LearnerStore] = None):
        self.runtime = runtime
        self.params = bkt_params or BKTParameters()
        self.store = store
        self.grounding_tool = OntologyGroundingTool(runtime)
        self.bkt_tool = KnowledgeTracingTool(store, self.params)
        self.ranking_tool = RecommendationRankingTool(runtime)

    @classmethod
    def from_config(cls, config_path: str | Path, model_path: Optional[str | Path] = None, store_path: Optional[str | Path] = None) -> HLORecommenderAgent:
        rt = Runtime.from_file(config_path)
        params = load_parameters(model_path) if model_path else BKTParameters()
        store = LearnerStore(store_path, params) if store_path else None
        return cls(rt, params, store)

    def record_assessment(self, event: dict) -> dict:
        if not self.store:
            raise RuntimeError("LearnerStore is not initialized for this agent.")
        self.store.observe(event)
        return self.store.snapshot(event["learner_id"])

    def consult(self, question: str, context: dict, learner_id: str = "default_learner") -> dict:
        if context.get("level") not in LEVELS:
            raise ValueError(f"Invalid learner level. Must be one of {LEVELS}")
        if not isinstance(context.get("minutes"), int) or context["minutes"] <= 0:
            raise ValueError("Available time (minutes) must be a positive integer.")

        # Step 1: Ontology Grounding
        ground_res = self.grounding_tool.execute(question)
        if ground_res.status != "success":
            return {
                "status": "clarification_required",
                "message": ground_res.message,
                "target_skills": [],
                "curriculum": [],
                "recommendations": []
            }

        target_skills = [s["uri"] for s in ground_res.data]

        # Step 2: Knowledge Tracing via BKT
        bkt_res = self.bkt_tool.execute(learner_id, target_skills)
        learner_state = {s: d["mastery"] for s, d in bkt_res.data.items()}

        # Step 3: Candidate Retrieval & Adaptive Ranking
        rank_res = self.ranking_tool.execute(question, context, state=learner_state)
        rec_data = rank_res.data

        # Step 4: Pedagogical Plan Synthesis
        curriculum = []
        for r in rec_data.get("ranked", []):
            curriculum.append({
                "resource_id": r["resource_id"],
                "title": r["title"],
                "score": round(r["score"], 4),
                "duration_minutes": r["features"]["time"],
                "mastery_gap": round(r["features"]["gap"], 4),
                "readiness": round(r["features"]["readiness"], 4),
                "rationale": self._generate_rationale(r)
            })

        return {
            "status": "success",
            "learner_id": learner_id,
            "clinical_intent": question,
            "context": context,
            "ontology_grounding": ground_res.data,
            "learner_mastery_diagnosis": bkt_res.data,
            "curriculum": curriculum,
            "total_eligible_resources": len(curriculum)
        }

    def _generate_rationale(self, ranked_item: dict) -> str:
        gap_val = ranked_item["features"]["gap"]
        readiness_val = ranked_item["features"]["readiness"]
        if gap_val > 0.7:
            return "High knowledge gap detected; selected for foundational core competency reinforcement."
        if readiness_val > 0.8:
            return "Learner prerequisites satisfied; optimal candidate for progressive case practice."
        return "Balanced resource matching clinical level and allocated study duration."
