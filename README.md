# README
Hi, this is just some experimentation with the Kaggle credit card fraud dataset https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

## What I did
I do some data exploration in the exploration/exploration.ipynb notebook and then I go to train some models in the src/running_models/train_model.py script.

This was a good way for me to get to grips with dealing with an imbalanced dataset.

## Where to put the data
To run any of this you neeed to download the dataset from the link at the top and put it under data so that it reads data/creditcard.csv

## How to Run Training Script
Once you've got the data in the right place you can train and test your model by:
- Install uv if you haven't already (Documentation [here][uv_doc_link])
- Use ```uv sync``` at the root of the directory (this will install the necessary packages)
- You should then activate your environment. (```source .venv/Scripts/activate```) 
- cd into src/running_models
- you should be able to run  ```python train_model.py```

**Note**: Figures should save to data/outputs so if you haven't made this file the code will.

## Room For Improvement
- There is no feature seletion in this process which probably hampers performance. A proper solution would attempt to select appropriate features.
- Likewise there is no model selection process aside from me selecting between two methods of dealing with the data imbalance in the data.

[uv_doc_link]: https://docs.astral.sh/uv/getting-started/installation/
