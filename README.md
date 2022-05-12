# SemEval-2022 Task 09: R2VQ - Competence-based Multimodal Question Answering

This code repository implements the winning solution for SemEval 2022 Task 9  
https://competitions.codalab.org/competitions/34056  
and it is related to our research paper

**Samsung Research Poland (SRPOL) at SemEval-2022 Task 9:  
Hybrid Question Answering Using Semantic Roles**

(BibTex entry will be updated after the conference proceedings are published).

## Setup
```
git submodule init
git submodule update
virtualenv -p `which python3.8` ./venv
source ./venv/bin/activate
pip install -r ./requirements.txt
./run_tests.sh
```

## Launch
```
PYTHONPATH=`pwd` ./bin/run_end_to_end_prediction.py --which (train|val|test)
```
The results will appear in `results` and in `results/per_category/(train|val|test)/`.

Detailed instructions can be found in separate READMEs in subfolders
