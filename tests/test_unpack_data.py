import unittest

from src.get_root import get_root
from src.unpack_data import Recipe, Q_A, convert_dataset, rewrite_to_list_of_questions, QuestionAnswerRecipe


class UnpackData(unittest.TestCase):

    def test_convert_train_data(self):
        filename = f"{get_root()}/modules/recipe2video/data/train/crl_srl.csv"
        dataset = convert_dataset(filename, use_tqdm=False, limit_recipes=1)
        self.assertIsInstance(dataset, list)
        self.assertEqual(1, len(dataset))
        all_are_recepies = [isinstance(x, Recipe) for x in dataset]
        self.assertTrue(all(all_are_recepies), msg=f"actual = {all_are_recepies}")

        first = dataset[0]
        self.assertIsInstance(first, Recipe)
        self.assertEqual("f-6VWP66LZ", first.id)
        self.assertRegex(first.new_pars_str.replace("\n", " ## "), "cut carrots and zucchini into cubes")
        self.assertEqual(27, len(first.q_a))
        self.assertIsInstance(first.q_a[0], Q_A)
        self.assertEqual("How many actions does it take to process the minced meat?", first.q_a[0].q)


    def test_rewrite_to_list_of_qars(self):
        filename = f"{get_root()}/modules/recipe2video/data/train/crl_srl.csv"
        list_of_recs = convert_dataset(filename, use_tqdm=False, limit_recipes=2)
        as_qars = rewrite_to_list_of_questions(list_of_recs)
        self.assertTrue(all([isinstance(x, QuestionAnswerRecipe) for x in as_qars]))
        self.assertEqual(27 + 57, len(as_qars))

        first = as_qars[0]
        self.assertEqual("How many actions does it take to process the minced meat?", first.question)
        self.assertEqual("1", first.answer)
        self.assertEqual("0-1", first.question_class)
        self.assertRegex(first.recipe_passage, "In a separate pan saute minced meat breaking it up well")
        self.assertNotRegex(first.recipe_passage, "# text = ")

        self.assertIsInstance(first.qa_copy, Q_A)
        self.assertIsInstance(first.recipe, Recipe)
