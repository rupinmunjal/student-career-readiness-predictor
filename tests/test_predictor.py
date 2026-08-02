import unittest

import pandas as pd

from src import StudentCareerWillingnessPredictor


class TestStudentCareerWillingnessPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = StudentCareerWillingnessPredictor()
        cls.predictor.train("data/Student Attitude and Behavior.csv")

    def test_single_prediction_profile(self) -> None:
        profile = self.predictor.predict_profile(
            {
                "certification_course": 1.0,
                "tenth_mark": 82.0,
                "twelfth_mark": 78.0,
                "college_mark": 80.0,
                "daily_studying_time": 3.0,
                "likes_degree": 1.0,
                "travel_time": 1.0,
                "stress_level": 2.0,
                "part_time_job": 0.0,
                "github_activity_proxy": 6.0,
                "side_projects_count": 4.0,
                "internship_experience": 1.0,
                "tech_skills_self_rating": 7.0,
                "languages_known_count": 4.0,
            }
        )
        self.assertGreaterEqual(profile["readiness_score"], 0.0)
        self.assertLessEqual(profile["readiness_score"], 100.0)
        self.assertIn(profile["best_domain"], {"Web Dev", "Data", "DevOps", "Mobile"})
        self.assertEqual(len(profile["confidence_interval"]), 2)
        self.assertEqual(set(profile["domain_scores"].keys()), {"Web Dev", "Data", "DevOps", "Mobile"})

    def test_batch_prediction_output_columns(self) -> None:
        batch = pd.DataFrame(
            [
                {
                    "tenth_mark": 85,
                    "twelfth_mark": 79,
                    "college_mark": 81,
                    "certification_course": 1,
                    "daily_studying_time": 3,
                    "likes_degree": 1,
                    "travel_time": 2,
                    "stress_level": 1,
                    "part_time_job": 0,
                    "github_activity_proxy": 6,
                    "side_projects_count": 4,
                    "internship_experience": 1,
                    "tech_skills_self_rating": 7,
                    "languages_known_count": 4,
                },
                {
                    "tenth_mark": 60,
                    "twelfth_mark": 58,
                    "college_mark": 62,
                    "certification_course": 0,
                    "daily_studying_time": 1,
                    "likes_degree": 0,
                    "travel_time": 4,
                    "stress_level": 3,
                    "part_time_job": 1,
                    "github_activity_proxy": 2,
                    "side_projects_count": 1,
                    "internship_experience": 0,
                    "tech_skills_self_rating": 4,
                    "languages_known_count": 2,
                },
            ]
        )
        out = self.predictor.predict_batch(batch)
        for col in [
            "predicted_junior_readiness",
            "confidence_low",
            "confidence_high",
            "best_domain",
            "domain_web_dev",
            "domain_data",
            "domain_devops",
            "domain_mobile",
        ]:
            self.assertIn(col, out.columns)
        self.assertEqual(len(out), 2)

    def test_batch_prediction_with_missing_columns_uses_defaults(self) -> None:
        partial = pd.DataFrame(
            [
                {"tenth_mark": 80, "twelfth_mark": 75, "college_mark": 78},
                {"tenth_mark": 55, "college_mark": 60},
            ]
        )
        out = self.predictor.predict_batch(partial)
        self.assertEqual(len(out), 2)
        for col in ["predicted_junior_readiness", "best_domain", "readiness_level"]:
            self.assertIn(col, out.columns)
        self.assertTrue((out["predicted_junior_readiness"] >= 0).all())
        self.assertTrue((out["predicted_junior_readiness"] <= 100).all())

    def test_skill_gap_analysis(self) -> None:
        gap = self.predictor.skill_gap_analysis(
            {
                "certification_course": 0,
                "tenth_mark": 68,
                "twelfth_mark": 66,
                "college_mark": 64,
                "daily_studying_time": 1.0,
                "likes_degree": 0,
                "travel_time": 4.0,
                "stress_level": 3.0,
                "part_time_job": 1,
                "github_activity_proxy": 2.0,
                "side_projects_count": 1.0,
                "internship_experience": 0.0,
                "tech_skills_self_rating": 4.0,
                "languages_known_count": 1.0,
            },
            target_role="Data",
        )
        self.assertEqual(gap["target_role"], "Data")
        self.assertGreaterEqual(len(gap["missing"]), 1)


if __name__ == "__main__":
    unittest.main()