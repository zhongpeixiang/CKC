
#### This repo contains the code for our work published in AAAI 2021: 

`Keyword-Guided Neural Conversational Model`

This repo is mainly adapted from https://github.com/squareRoot3/Target-Guided-Conversation.

### Datasets
- ConvAI2: https://www.dropbox.com/s/1fw2gwpuyud2bkq/convai2.zip?dl=0
- Reddit/Casual: https://www.dropbox.com/s/hhesm3z6judpsxa/casual.zip?dl=0

### Steps:
1. Download the datasets.
2. Run `python train_keyword_prediction.py --config ./configs/keyword_prediction_config.json` to train a keyword prediction model.
3. Run `python train_retrieval.py --config ./configs/retrieval_config.json` to train a response retrieval model.

Note that this repo contains a portion of experimental code that is not used in the paper. 