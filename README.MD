# GPT FROM SCRATCH
This code was created following the workshop "Building LLMs from the Ground Up: A 3-hour Coding Workshop" given by Sebastian Raschka. 
Here you can find the code needed to train an LLM with your own data without using pre-trained models.

## Requirements
- Python >= 3.8
- Conda/venv
- Txt file with all your training text
- Run command `pip install -r requirements.txt` to install all the necessary packages (it's highly recommended to use a virtual environment as conda, mamba, venv, etc...).

## How to train
1. Replace all the text of `dataset.txt` file that is in the root of the project.
2. Call the training script as `python ./train.py`

## How to test
1. In the script `test.py`, line 22, select the model generated after the training.
2. Line 24: write the "prompt" text.
3. Line 26: select the number of tokens to generate in the inference modifying the parameter `max_new_tokens`
4. Call the test script as `python ./test.py`