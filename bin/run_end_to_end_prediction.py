#!/bin/env python3
#
#  Call me:
#  PYTHONPATH=`pwd` ./bin/run_end_to_end_prediction.py  --which [train|test|val]
#
#

import argparse
import os.path

from src.fetch_resources import fetch_linguistic_resources

from src.get_root import get_root
from src.pipeline.answerers.class_0 import QuestionAnswerer0
from src.pipeline.answerers.class_0_how_many_times import QuestionAnswerer0HowManyTimes
from src.pipeline.answerers.class_0_how_many_actions import QuestionAnswerer0HowManyActions
from src.pipeline.answerers.class_1_v2 import QuestionAnswerer1Class8Style
from src.pipeline.answerers.class_3_what import QuestionAnswerer3What
from src.pipeline.answerers.class_14_preheat import QuestionAnswerer14Preheat
from src.pipeline.answerers.class_17 import QuestionAnswerer17
from src.pipeline.answerers.class_2_6_10_14 import QuestionAnswerer2_6_10_14
from src.pipeline.answerers.class_2_where import QuestionAnswerer2Where
from src.pipeline.answerers.class_4_v2 import QuestionAnswerer4EventBased
from src.pipeline.answerers.class_8 import QuestionAnswerer8
from src.pipeline.answerers.class_3_how import QuestionAnswerer3How
from src.pipeline.deterministic_qa_engine import QuestionAnswererNA
from src.pipeline.end_to_end_prediction import EndToEndQuestionAnsweringPrediction
from src.pipeline.extractive_qa import ExtractiveQuestionAnswererFactory
from src.pipeline.handler_metrics import HandlerF1, HandlerExactMatch
from src.pipeline.handler_metrics_per_category import HandlerMetricsPerCategory
from src.pipeline.question_answering_dispatcher import QuestionAnsweringDispatcher


def get_dispatching_engine() -> QuestionAnsweringDispatcher:
    dispatching_rules = {
        "0_times": QuestionAnswerer0HowManyTimes(),
        "0_actions": QuestionAnswerer0HowManyActions(),
        "0_are_used": QuestionAnswerer0(),
        "1": QuestionAnswerer1Class8Style(["HABITAT", "TOOL"], ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"], ["Drop"]),
        "2_where":QuestionAnswerer2Where(),
        "2_6_10_14": QuestionAnswerer2_6_10_14(
            ["EXPLICITINGREDIENT", "IMPLICITINGREDIENT"], ["TOOL"],
            ["Patient", "Theme"], ["Instrument"],
            ["Patient", "Theme"], ["Attribute"],
            ["Patient", "Theme"], ["Goal"]
        ),
        "3_how": QuestionAnswerer3How(),
        "3_what": QuestionAnswerer3What(),
        "4": QuestionAnswerer4EventBased(),
        "5": QuestionAnswerer8(["Patient"], ["Result"]),
        "7": QuestionAnswerer8(["Patient", "Attribute", "Purpose"], ["Time"], reversed_paragraphs=False),
        "8": QuestionAnswerer8(["Patient", "Theme"], ["Location", "Destination", "Co-Patient", "Co-Theme"]),
        "9": QuestionAnswerer8(["Patient"], ["Extent"]),
        "11_15": QuestionAnswerer8(["Patient"], ["Cause", "Purpose"]),
        "12_13": QuestionAnswerer8(["Patient", "Theme"], ["Co-Patient", "Co-Theme"]),
        "14_preheat": QuestionAnswerer14Preheat(),
        "16": QuestionAnswerer8(["Patient"], ["Source"], reversed_paragraphs=False),
        "17": QuestionAnswerer17(),
        "18": ExtractiveQuestionAnswererFactory.get_extractive_answerer(),
        "RC": ExtractiveQuestionAnswererFactory.get_extractive_answerer()
    }

    # Leave empty to run all answerers.
    # If nonempty, this will answer only for given categories.
    spare_categories = {}
    if spare_categories:
        for k in dispatching_rules.keys():
            if k not in spare_categories:
                dispatching_rules[k] = QuestionAnswererNA()

    return QuestionAnsweringDispatcher(dispatching_rules)


def launch(parsed_args: argparse.Namespace) -> None:
    fetch_linguistic_resources()

    ExtractiveQuestionAnswererFactory.set_default_engine(parsed_args.which)

    engine = EndToEndQuestionAnsweringPrediction(parsed_args.which, parsed_args.with_postprocessing,
                                                 get_dispatching_engine())
    engine.limit_recipes = None
    engine.use_tqdm = True
    # append custom post processor handlers here:
    engine.add_qa_handler(HandlerF1())
    engine.add_qa_handler(HandlerExactMatch())
    prefix = os.path.join(get_root(), "results", "per_category", parsed_args.which)
    engine.add_qa_handler(HandlerMetricsPerCategory(prefix))

    more_info = {"use_tqdm": True}
    questions, answers = engine.run_prediction(more_info)
    print(f"len Qs = {len(questions)}")
    print(f"len As = {len(answers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", type=str, default="train", choices={"train", "test", "val"},
                        help="Which dataset should be used (train / test/ val). " \
                             "Note that 'test' doesn't contain answers!")
    parser.add_argument("--with_postprocessing", action='store_true',
                        help="Add this argument if need Bert NA postprocessing on val and test set")
    parsed_args = parser.parse_args()

    launch(parsed_args)
