# Reading Comprehension module
Reading Comprehension module used for extractive question answering.

## Setup
Install repository requirements and RC specific requirements.
```shell script
pip install -r requirements.txt
pip install -r src/reading_comprehension/requirements.txt
```

## Usage
Use a model fine-tuned on SQuAD 2.0 and train it further on semeval dataset. Example use: 

```
./train.sh \
    --model=ahotrod/electra_large_discriminator_squad2_512  \
    --include_ingredients \
    --gpu=0,1,2,3
```

To get predictions run `predict.sh`. To run end-to-end prediction pipeline and get EM and F1 score per created class,
use `--end2end` argument. Example use:
```
./predict.sh \
    --end2end \
    --which=val \
    --model=results/ahotrod_electra_large_discriminator_squad2_512_include_ingredients \
    --epoch=5 \
    --include_ingredients \
    --gpu=0,1,2,3
```

For parameters and hyperparameters refer to Yaml files in `src/reading_comprehension/configs`.
