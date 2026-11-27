import unittest
from dataclasses import replace
from pathlib import Path
from hlo_recommender.agent import HLORecommenderAgent
from hlo_recommender.models.bkt import BKTParameters, LearnerStore
from hlo_recommender.pipeline import Runtime


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.runtime = Runtime.from_file(self.root / "configs/default.toml")
        self.params = BKTParameters(initial=0.2, learn=0.1, guess=0.2, slip=0.1)

    def test_agent_grounding_and_consult(self):
        agent = HLORecommenderAgent(self.runtime, self.params)
        self.runtime.config["top_k"] = 1
        res = agent.consult(
            question="Find resources for arterial blood gas interpretation.",
            context={"level": "Junior", "minutes": 15},
            learner_id="test_learner"
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["curriculum"]), 1)
        self.assertEqual(res["total_eligible_resources"], 4)
        self.assertIn("learner_mastery_diagnosis", res)
        self.assertIn("ontology_grounding", res)
        for item in res["curriculum"]:
            self.assertIn("rationale", item)
            self.assertIn("score", item)
            self.assertEqual(item["duration_minutes"], self.runtime.by_id[item["resource_id"]].minutes)

    def test_grounded_agent_question_with_no_eligible_resource(self):
        agent = HLORecommenderAgent(self.runtime, self.params)
        result = agent.consult(
            "Find resources for arterial blood gas interpretation.",
            {"level": "Junior", "minutes": 1},
        )
        self.assertEqual(result["status"], "no_eligible_resource")
        self.assertEqual(result["curriculum"], [])
        self.assertEqual(result["total_eligible_resources"], 0)

    def test_agent_uses_model_prior_for_unobserved_mastery(self):
        params = BKTParameters(initial=0.85)
        agent = HLORecommenderAgent(self.runtime, params)
        result = agent.consult(
            "Find resources for arterial blood gas interpretation.",
            {"level": "Senior", "minutes": 20},
        )
        self.assertEqual(result["total_eligible_resources"], 3)
        for item in result["curriculum"]:
            self.assertAlmostEqual(item["mastery_gap"], 0.15)

    def test_agent_uses_observed_mastery_for_prerequisites_outside_question(self):
        question = "Find resources for arterial blood gas interpretation."
        context = {"level": "Junior", "minutes": 8}
        resource_id = "https://example.org/hlo-research/resource/K002-1"
        prerequisite = "http://example.org/hlo.owl#K001"
        resource = self.runtime.by_id[resource_id]
        self.runtime.by_id[resource_id] = replace(resource, prerequisites=(prerequisite,))
        store = LearnerStore(":memory:", self.params)
        self.addCleanup(store.close)
        agent = HLORecommenderAgent(self.runtime, self.params, store)
        for i in range(2):
            agent.record_assessment({
                "event_id": f"assessment-{i}",
                "learner_id": "learner",
                "skill_id": prerequisite,
                "correct": True,
                "timestamp": f"2026-11-27T10:0{i}:00+00:00",
                "data_status": "synthetic",
            })
        result = agent.consult(question, context, learner_id="learner")
        self.assertIn(resource_id, {item["resource_id"] for item in result["curriculum"]})
        self.assertNotIn(prerequisite, result["learner_mastery_diagnosis"])

    def test_agent_unrecognized_query(self):
        agent = HLORecommenderAgent(self.runtime, self.params)
        res = agent.consult(
            question="Random unrelated topic not in medical ontology",
            context={"level": "Junior", "minutes": 15},
            learner_id="test_learner"
        )
        self.assertEqual(res["status"], "clarification_required")
        self.assertEqual(len(res["curriculum"]), 0)


if __name__ == "__main__":
    unittest.main()
