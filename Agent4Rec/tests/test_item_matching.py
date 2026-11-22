import unittest

from simulation.item_matching import match_title_to_candidates, normalize_title


class ItemMatchingTest(unittest.TestCase):
    def setUp(self):
        self.ids = [282, 1643, 957, 319]
        self.titles = [
            "Shawshank Redemption, The (1994)",
            "Saving Private Ryan (1998)",
            "English Patient, The (1996)",
            "Forrest Gump (1994)",
        ]

    def test_article_and_year_normalization(self):
        self.assertEqual(normalize_title("The Shawshank Redemption"),
                         normalize_title(self.titles[0]))
        match = match_title_to_candidates(
            "The Shawshank Redemption", self.ids, self.titles
        )
        self.assertEqual(match.item_id, 282)
        self.assertEqual(match.match_method, "normalized_exact")

    def test_candidate_constrained_fuzzy_match(self):
        match = match_title_to_candidates(
            "Saving Private Rian", self.ids, self.titles
        )
        self.assertEqual(match.item_id, 1643)
        self.assertEqual(match.match_method, "candidate_fuzzy")

    def test_unrelated_title_is_not_forced(self):
        match = match_title_to_candidates("Toy Story", self.ids, self.titles)
        self.assertIsNone(match.item_id)
        self.assertEqual(match.match_method, "unresolved")

    def test_ambiguous_duplicate_is_not_forced(self):
        match = match_title_to_candidates(
            "King Kong", [10, 11], ["King Kong (1933)", "King Kong (1976)"]
        )
        self.assertIsNone(match.item_id)


if __name__ == "__main__":
    unittest.main()
