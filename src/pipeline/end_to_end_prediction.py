from typing import List, Dict, Any, Tuple

from src.get_root import get_root
from src.pipeline.handlers import InterfaceHandler, HandlerSaveToJson
from src.pipeline.question_answering_dispatcher import QuestionAnsweringDispatcher, PredictedAnswer
from src.unpack_data import QuestionAnswerRecipe, convert_train_data, convert_val_data, convert_test_data, \
    rewrite_to_list_of_questions


class EndToEndQuestionAnsweringPrediction:

    def __init__(self, which_dataset: str, with_postprocessing: bool,
                 dispatching_engine: QuestionAnsweringDispatcher = None, output_json_filename: str = None):
        if which_dataset not in {"train", "val", "test"}:
            raise ValueError(f"Incorrect dataset spec = {which_dataset}")
        self.which_dataset = which_dataset
        self.with_postprocessing = with_postprocessing
        self.dispatching_engine = dispatching_engine if dispatching_engine \
            else EndToEndQuestionAnsweringPrediction.get_dispatching_engine()

        self.output_json_filename = output_json_filename if output_json_filename \
            else f"{get_root()}/results/r2vq_pred__SRPOL_{which_dataset}.json"  # TODO add timestamp
        self.qa_handlers: List[InterfaceHandler] = []
        self.limit_recipes: int = None
        self.use_tqdm = False
        self.add_qa_handler(HandlerSaveToJson(self.output_json_filename))

    def add_qa_handler(self, a_handler) -> None:
        self.qa_handlers.append(a_handler)

    @staticmethod
    def get_dispatching_engine() -> QuestionAnsweringDispatcher:
        return QuestionAnsweringDispatcher()

    def load_dataset(self, limit_recipes: int = None) -> List[QuestionAnswerRecipe]:
        """
        :param limit_recipes:  break after this recipe (default = None = no limit)
        :return: the loaded dataset or part of it
        """
        loaders = {
            "train": convert_train_data,
            "val": convert_val_data,
            "test": convert_test_data
        }
        list_of_recipes = loaders[self.which_dataset](self.use_tqdm, limit_recipes=limit_recipes)
        return rewrite_to_list_of_questions(list_of_recepies=list_of_recipes)

    def run_prediction(self, more_info: Dict[str, Any] = {}) \
            -> Tuple[List[QuestionAnswerRecipe], List[PredictedAnswer]]:
        questions = self.load_dataset(self.limit_recipes)
        predicted_answers = self.dispatching_engine.predict_answers(self.which_dataset, self.with_postprocessing,
                                                                    questions, more_info)

        for handler in self.qa_handlers:
            handler.handle_questions_answers(questions, predicted_answers, more_info)

        return questions, predicted_answers
