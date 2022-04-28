import copy
from io import open
from random import randint
from typing import List, Dict, Optional

from conllu import parse, TokenList
from tqdm import tqdm

from src.get_root import get_root
from src.annotated_recipe import AnnotatedRecipe


class Q_A:
    def __init__(self, question: str, answer: str = ''):
        self.id = question.split('question ')[1].split(' =')[0]
        self.q = question.split(' = ')[1].strip()
        self.a = answer.split(' = ')[1].strip() if answer else None

        self.q_type = question.split("# question ")[1].split("-")[0]
        self.q_sub_type = question.split("# question ")[1].split("-")[1].split(' =')[0]

    @staticmethod
    def build_dummy_qa(question: str, category: str, answer: str = None):
        x = randint(0, 20)
        q = f'# question {category}-{x} = {question}'
        a = f'# answer {category}-{x} = {answer}' if answer else None
        return Q_A(q, a)



class Recipe:
    def __init__(self, recipe_raw: List[str]):
        self.id: str = recipe_raw[0].split(' = ')[1].strip()
        self.q_a: List[Q_A] = self._return_q_and_a(recipe_raw)
        self.q_a_str: str = self._return_q_and_a_str(recipe_raw)
        self.metadata: Dict[str, str] = self._return_metadata(recipe_raw)
        self.metadata_str: str = self._return_metadata_str(recipe_raw)
        id_new_pars_start = self._recipe_conllu_start(recipe_raw)
        self.new_pars: Dict[str, List[TokenList]] = self._return_new_pars(recipe_raw[id_new_pars_start:])
        self.new_pars_str: str = ''.join(recipe_raw[id_new_pars_start:])
        self.steps_str: str = self.return_recipe_steps()
        self.annotated_recipe = AnnotatedRecipe.parse_recipe_from_lines(recipe_raw)

    @staticmethod
    def return_recipe_for_test():
        data_path = f'{get_root()}/data/small_data/recipe.csv'
        with open(data_path, 'r', encoding='utf-8') as f:
            test_item = list(f)
        return Recipe(test_item)

    def return_recipe_steps(self) -> str:
        temp_str = ''
        for k, v in self.new_pars.items():
            if 'step' in k:
                temp_str += v[0].metadata['text'] + '\n'
        return temp_str

    @staticmethod
    def _recipe_conllu_start(recipe_raw: List[str]):
        for i, val in enumerate(recipe_raw):
            if 'newpar id' in val:
                return i

    @staticmethod
    def _return_q_and_a(recipe_raw: List[str]) -> List[Q_A]:
        q_a_to_return = []
        prev_line = ""
        for current_line in recipe_raw[1:]:
            if 'question' not in current_line and 'answer' not in current_line:
                break
            if "# answer" in current_line:
                assert "# question" in prev_line
                q_a_to_return.append(Q_A(prev_line, current_line))  # append answer and question
                prev_line = ""
            elif "# question" in prev_line:
                assert "# question" in current_line
                q_a_to_return.append(Q_A(prev_line))  # append question w/o answer
                prev_line = current_line
            elif prev_line == "":
                prev_line = current_line  # wait for the next line
            else:
                raise AssertionError(f"Shouldn't reach here: prev_line = {prev_line}, current_line = {current_line}")
        if "# question" in prev_line:
            q_a_to_return.append(Q_A(prev_line))  # append the last question

        return q_a_to_return

    @staticmethod
    def _return_q_and_a_str(recipe_raw: List[str]) -> str:
        temp = []
        for q_or_a in recipe_raw[1:]:
            if 'question' not in q_or_a and 'answer' not in q_or_a:
                break
            temp.append(q_or_a)
        return ''.join(temp)

    @staticmethod
    def _return_metadata(recipe_raw: List[str]) -> Dict[str, str]:
        temp = {}
        for r in recipe_raw:
            if 'metadata' in r:
                temp_metadata = r.split(' = ')
                key = temp_metadata[0].split('metadata:')[1]
                value = temp_metadata[1][:-1]
                temp[key] = value
        return temp

    @staticmethod
    def _return_metadata_str(recipe_raw: List[str]) -> str:
        temp = []
        for r in recipe_raw:
            if 'metadata' in r:
                temp.append(r)
        return ''.join(temp)

    def _return_new_pars(self, recipe_raw: List[str]) -> Dict[str, List[TokenList]]:
        temp = {}
        try:
            key = recipe_raw[0].split(f'{self.id}::')[1].strip()
        except Exception:
            print(f"line = {recipe_raw[0]} // split = {f'{self.id}::'}")
            raise

        val = []
        for r in recipe_raw[1:]:
            if 'newpar id' in r:
                temp[key] = parse(''.join(val))
                key = r.split(f'{self.id}::')[1][:-1]
                val = []
                continue
            val.append(r)
        temp[key] = parse(''.join(val))
        return temp


def convert_train_data(use_tqdm: bool = True, limit_recipes=None) -> List[Recipe]:
    data_path = f'{get_root()}/data/r2vq_train_10_28_2021/train/crl_srl.csv'
    return convert_dataset(data_path, use_tqdm, limit_recipes)


def convert_val_data(use_tqdm: bool = True, limit_recipes=None) -> List[Recipe]:
    data_path = f'{get_root()}/data/r2vq_val_12_03_2021/val/crl_srl.csv'
    return convert_dataset(data_path, use_tqdm, limit_recipes)


def convert_test_data(use_tqdm: bool = True, limit_recipes=None) -> List[Recipe]:
    data_path = f'{get_root()}/data/r2vq_test_12_03_2021/test/crl_srl.csv'
    return convert_dataset(data_path, use_tqdm, limit_recipes)


def convert_dataset(data_path: str, use_tqdm: bool, limit_recipes=None) -> List[Recipe]:
    recipes = []
    recipe = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = list(f)

    iterator = tqdm(data, desc="parsing") if use_tqdm else data
    for d in iterator:
        if recipe and 'newdoc id' in d:
            recipes.append(Recipe(recipe))
            recipe = []
        recipe.append(d)
        if limit_recipes and len(recipes) >= limit_recipes:
            return recipes
    recipes.append(Recipe(recipe))
    return recipes


class QuestionAnswerRecipe:

    def __init__(self, qa: Q_A, recipe: Recipe):
        self.recipe: Recipe = recipe
        self.qa_copy: Q_A = copy.copy(qa)
        self.question: str = qa.q
        self.answer: Optional[str] = qa.a if qa.a else None
        self.question_class: str = qa.id
        self.recipe_passage = QuestionAnswerRecipe.extract_recipe_only(recipe)

    @staticmethod
    def extract_recipe_only(recipe: Recipe) -> str:
        as_lines = recipe.new_pars_str.split("\n") if recipe else []
        lines_of_recipe = [line for line in as_lines if line.find("# text = ") == 0]
        lines_of_recipe = [line.split("=")[1] for line in lines_of_recipe]
        return "\n".join(lines_of_recipe)


def rewrite_to_list_of_questions(list_of_recepies: List[Recipe]) -> List[QuestionAnswerRecipe]:
    ret = []
    for recipe in list_of_recepies:
        for qa in recipe.q_a:
            ret.append(QuestionAnswerRecipe(qa, recipe))
    return ret


if __name__ == "__main__":
    question = 'What do you serve the vegetables and minced meat casserole with?'
    answer = '# answer 13-0 = with cooked rice or something similar ( couscous , mashed potatoes )'
    q_a_test = Q_A.build_dummy_qa(question, "13", answer)
    recipe_test = Recipe.return_recipe_for_test()
    train_data = convert_train_data()
    val_data = convert_val_data()
    test_data = convert_test_data()
    print("done")
