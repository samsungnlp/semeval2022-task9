import unittest

from src.annotated_recipe import AnnotatedToken, AnnotatedSentence, AnnotatedRecipe
from src.get_root import get_root


class TestAnnotatedToken(unittest.TestCase):

    def test_parse(self):
        a_line = "3\tserving\tserving\tNOUN\tO\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_\t_"
        res = AnnotatedToken.parse_from_line(a_line)
        self.assertIsInstance(res, AnnotatedToken)
        self.assertEqual(3, res.id)
        self.assertEqual("serving", res.raw_token)
        self.assertEqual("serving", res.normalized_token)
        self.assertEqual("NOUN", res.part_of_speech)

        self.assertEqual("O", res.role_in_recipe)
        self.assertIsNone(res.where_is_my_verb_implicit)
        self.assertIsNone(res.where_is_my_verb_explicit)
        self.assertEqual(2, res.position_in_the_whole_recipe)

    def test_parse_with_extra_info(self):
        a_line = "18\tzucchini\tzucchini\tNOUN\tB-EXPLICITINGREDIENT\t15\t_\t_\tzucchini.1.1.18\t_\t_\tI-Patient" \
                 "\t_\t_\t_\t_\t_\t_\t_\t_"
        res = AnnotatedToken.parse_from_line(a_line, 20)
        self.assertIsInstance(res, AnnotatedToken)
        self.assertEqual(18, res.id)
        self.assertEqual("zucchini", res.raw_token)
        self.assertEqual("zucchini", res.normalized_token)
        self.assertEqual("NOUN", res.part_of_speech)

        self.assertEqual("B-EXPLICITINGREDIENT", res.role_in_recipe)
        self.assertEqual(15, res.where_is_my_verb_explicit)
        self.assertIsNone(res.where_is_my_verb_implicit)
        self.assertEqual("zucchini.1.1.18", res.relation2)

        self.assertEqual(37, res.position_in_the_whole_recipe)  # note: counted from 0 rather than 1
        self.assertIn("I-Patient", res.semantic_roles)


class TestAnnotatedSentence(unittest.TestCase):

    def test_parse_sentence_from_lines(self):
        lines = [
            "# text = Cut the ",
            "# newpar id = f-6VWP66LZ::step01",
            "# sent_id = f-6VWP66LZ::step01::sent01",
            "1\tCut\tcut\tVERB\tB-EVENT\t_\t_\tResult=chopped_vegetables.1.1.1|Tool=knife.1.1.1\t_\tCUT\tB-V\t_\t_\t_\t_\t_\t_\t_\t_\t_",
            "2\tthe\tthe\tDET\tO\t_\t_\t_\t_\t_\tB-Patient\t_\t_\t_\t_\t_\t_\t_\t_\t_"
        ]
        res = AnnotatedSentence.parse_sentence_from_lines(lines)

        self.assertIsInstance(res, AnnotatedSentence)
        self.assertEqual("Cut the", res.raw_sentence)
        self.assertEqual("f-6VWP66LZ::step01", res.paragraph_id)
        self.assertEqual("f-6VWP66LZ::step01::sent01", res.sentence_id)
        self.assertEqual(0, res.sentence_position_in_paragraph)
        self.assertEqual(2, len(res.annotated_tokens))
        self.assertEqual("Cut", res.annotated_tokens[0].raw_token)
        self.assertEqual("CUT", res.annotated_tokens[0].verb_group)

        self.assertEqual("the", res.annotated_tokens[1].raw_token)
        self.assertEqual("DET", res.annotated_tokens[1].part_of_speech)


class TestAnnotatedRecipe(unittest.TestCase):

    def test_parse_from_recipe(self):
        lines = [
            "# newdoc id = f-GGX2LSGX",
            "# question 4-2 = A, B which comes first?",
            "# answer 4-2 = the first event",
            "# metadata:url = https://www",
            "# metadata:num_steps = 7",
            "# metadata:avg_len_steps = 31",
            "# metadata:num_ingres = 10",
            "# metadata:cluster = 8",
            "# newpar id = f-GGX2LSGX::ingredients",
            "# sent_id = f-GGX2LSGX::ingredients::sent01",
            "# text = 4 cups yellow onions, thinly sliced",
            "1	4	4	NUM	_	_	_	_	_	_	_	_	_	_	_	_	_	_	_	_",
            "2	cups	cup	NOUN	_	_	_	_	_	_	_	_	_	_	_	_	_	_	_	_",
            "",
            "# newpar id = f-GGX2LSGX::step01",
            "# sent_id = f-GGX2LSGX::step01::sent01",
            "# text = Preheat the oven to 350deg F.",
            "1	Preheat	preheat	VERB	B-EVENT	_	_	_	_	HEAT	B-V	_	_	_	_	_	_	_	_	_",
            "3	oven	oven	NOUN	B-HABITAT	1	_	_	oven.1.1.3	_	I-Patient	_	_	_	_	_	_	_	_	_",
            "# newpar id = f-GGX2LSGX::step02",
            "# sent_id = f-GGX2LSGX::step02::sent01",
            "# text = Heat the olive oil and butter.",
            "1	Heat	heat	VERB	B-EVENT	_	_	_	_	HEAT	B-V	_	_	_	_	_	_	_	_	_",
        ]
        res = AnnotatedRecipe.parse_recipe_from_lines(lines)
        self.assertIsInstance(res, AnnotatedRecipe)

        self.assertEqual("https://www", res.url)
        self.assertEqual("f-GGX2LSGX", res.recipe_id)

        self.assertEqual(7, res.num_steps)
        self.assertEqual(10, res.num_ingredients)

        self.assertEqual(3, len(res.annotated_sentences))
        self.assertEqual("4 cups yellow onions, thinly sliced", res.annotated_sentences[0].raw_sentence)
        self.assertEqual("4", res.annotated_sentences[0].annotated_tokens[0].raw_token)
        self.assertEqual("cups", res.annotated_sentences[0].annotated_tokens[1].raw_token)

        self.assertEqual("Preheat the oven to 350deg F.", res.annotated_sentences[1].raw_sentence)
        self.assertEqual(1, res.annotated_sentences[1].sentence_position_in_paragraph)
        self.assertEqual("f-GGX2LSGX::step01::sent01", res.annotated_sentences[1].sentence_id)
        self.assertEqual("Preheat", res.annotated_sentences[1].annotated_tokens[0].raw_token)
        self.assertEqual("oven", res.annotated_sentences[1].annotated_tokens[1].raw_token)

        self.assertEqual("Heat the olive oil and butter.", res.annotated_sentences[2].raw_sentence)
        self.assertEqual(2, res.annotated_sentences[2].sentence_position_in_paragraph)
        self.assertEqual("f-GGX2LSGX::step02::sent01", res.annotated_sentences[2].sentence_id)
        self.assertEqual("Heat", res.annotated_sentences[2].annotated_tokens[0].raw_token)
        self.assertEqual(4, res.annotated_sentences[2].annotated_tokens[0].position_in_the_whole_recipe)

    def test_parsing_from_actual_resource(self):
        file = f"{get_root()}/modules/recipe2video/data/train/crl_srl.csv"
        with open(file) as f:
            lines = [l.strip() for l in f]
            lines = lines[:268]

        a_recipe = AnnotatedRecipe.parse_recipe_from_lines(lines)
        self.assertEqual("f-6VWP66LZ", a_recipe.recipe_id)
        self.assertEqual(15, len(a_recipe.annotated_sentences))
        first = a_recipe.annotated_sentences[0]
        self.assertEqual(2, len(first.annotated_tokens))
        self.assertEqual("500g", first.annotated_tokens[0].raw_token)
        self.assertEqual("broccoli", first.annotated_tokens[1].raw_token)

        last = a_recipe.annotated_sentences[-1]
        self.assertEqual(14, len(last.annotated_tokens))
        self.assertEqual("Serve", last.annotated_tokens[0].raw_token)
        self.assertEqual("with", last.annotated_tokens[1].raw_token)
        self.assertEqual(".", last.annotated_tokens[-1].raw_token)
        self.assertEqual("PUNCT", last.annotated_tokens[-1].part_of_speech)
