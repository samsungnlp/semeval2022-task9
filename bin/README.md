# Main prediction engine

## Setup

```
virtualenv -p `which python3.8` ./venv
source ./venv/bin/activate
pip install -r ./requirements.txt
```

## Usage

```
PYTHONPATH=`pwd` ./bin/run_end_to_end_prediction.py  --which [train|val|test]
```

Check the file: `results/r2vq_pred__SRPOL_[which].json`

## How to add your own classifier?

1. Goto `src/pipeline`
2. Create a new class for your classifier. It **must** derive from
    * `InterfaceQuestionAnswering` (need full implementation)
    * or `QuestionAnsweringBase` (some partial implementation available)
3. Implement an interface method

```
def answer_a_question(self, question: QuestionAnswerRecipe,
                      question_category: QuestionCategory,
                      more_info: Dict[str, Any] = {}) -> PredictedAnswer
```

and / or

```
def batch_answer_questions(self, questions: List[QuestionAnswerRecipe], 
                           categories: List[QuestionCategory],
                           more_info: Dict[str, Any] = {}) -> List[PredictedAnswer]:
```

4. Goto `bin/run_end_to_end_prediction.py`
5. Add your classifier into the dispatcher table in

```
def get_dispatching_engine() -> QuestionAnsweringDispatcher:
```

6. Run the main script

```
PYTHONPATH=`pwd` ./bin/run_end_to_end_prediction.py  --which [train|val|test]
```

7. Check your results in

```
   results/r2vq_pred__SRPOL_[which].json`
```

## How to add your own post-processor (e.g. results checker)?

1. Goto `src/pipeline`
2. Create a new class for your handler. It **must** derive from `InterfaceHandler`
3. Implement interface method

```
def handle_questions_answers(self, 
                             questions: List[QuestionAnswerRecipe],
                             answers: List[PredictedAnswer],
                             more_info: Dict[str, Any] = {}):
```

4. Goto `bin/run_end_to_end_prediction.py`
5. Append your handler **before invoking prediction**:

```
engine.add_qa_handler(DummyHandler())
```
6. Rerun the script. The handlers are called after the processing
