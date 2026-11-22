import unittest
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
        res = agent.consult(
            question="Find resources for arterial blood gas interpretation.",
            context={"level": "Junior", "minutes": 15},
            learner_id="test_learner"
        )
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["curriculum"]), 0)
        self.assertIn("learner_mastery_diagnosis", res)
        self.assertIn("ontology_grounding", res)
        for item in res["curriculum"]:
            self.assertIn("rationale", item)
            self.assertIn("score", item)

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
