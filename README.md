## About The Project

This is a review classifier based on sentiment analysis built under the specification for the second challenge of the contest [Reto Enseña Oracle](https://nuwe.io/dev/competitions/reto-ensena-oracle-espana/sentimiento-y-deploymentML), hence the generated model will be capable of identifying the positive or negative sentiment of a given review.

The main functionality of the project can be summarized as follows:

* Usage of predefined [BERT](https://doi.org/10.48550/arXiv.1810.04805), [distilBERT](https://doi.org/10.48550/arXiv.1910.01108) or [RoBERTa](https://doi.org/10.48550/arXiv.1907.11692) models.
* Transfer learning from pretrained, either:
  * `bert-base-uncased`.
  * `distilbert-base-uncased-finetuned-sst-2-english`.
  * `roberta-base`.
  * `cardiffnlp/twitter-roberta-base-sentiment-latest`.
  * `siebert/sentiment-roberta-large-english`.
  * `roberta-large-mnli`.
* Fine tune the model using the [provided dataset](https://storage.googleapis.com/challenges_events/03_2023/Oracle%202nd%20Reto/Data/train.csv).
  * Segregate a portion of the dataset for validation purposes and hence avoiding overfitting.
* Inference and result saving for test dataset as _json_ file.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built Using

Base technologies:

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Transformers](https://huggingface.co/docs/transformers/index)

Additional dependencies:

* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Sklearn](https://scikit-learn.org/stable/)

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

Given that [Python 3.9+](https://www.python.org/downloads/) and [pip](https://pypi.org/project/pip/) are installed and correctly configured in the system, and that you have [CUDA-capable hardware](https://developer.nvidia.com/cuda-gpus) installed, you may follow these steps.

> You will require at least 12GB of VRAM to load `robertav3` and `robertav4` models onto the GPU, and 6GB for the rest

### Prerequisites

* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 11.0 or above is correctly installed.
* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) version 7 or above is correctly installed.

### Installation

1. Clone this repository locally.

```bash
git clone git@github.com:sperezacuna/oracle-challenge-f2.git
```
2. Create Python [virtual environment](https://docs.python.org/3/library/venv.html) and activate it (**recommended**)

```bash
python -m venv env
source env/bin/activate 
```

3. Install all required dependencies.

```bash
pip install -r requirements.txt
```

## Execution

1. Train a new model based on the [provided train dataset](https://storage.googleapis.com/challenges_events/03_2023/Oracle%202nd%20Reto/Data/train.csv) using `generate_model.py` script. You may specify the following parameters:
    
    `-m MODELTYPE`, to establish the base classification model type, either:
    
      - `bert`, to train a [BERT](https://doi.org/10.48550/arXiv.1810.04805) model using transfer learning from [bert-base-uncased](https://huggingface.co/bert-base-uncased)
      - `distilbert`, to train a [distilBERT](https://doi.org/10.48550/arXiv.1910.01108) model using transfer learning from [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
      - `roberta`, to train a [RoBERTa](https://doi.org/10.48550/arXiv.1907.11692) model using transfer learning from [roberta-base](https://huggingface.co/roberta-base).
      - `robertav2`, to train a [RoBERTa](https://doi.org/10.48550/arXiv.1907.11692) model using transfer learning from [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
      - `robertav3`, to train a [RoBERTa](https://doi.org/10.48550/arXiv.1907.11692) model using transfer learning from [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english).
      - `robertav4`, to train a [RoBERTa](https://doi.org/10.48550/arXiv.1907.11692) model using transfer learningn from [roberta-large-mnli](https://huggingface.co/roberta-large-mnli). This modeltype has performed best so far.
      
    `--help`, to show the help message for the script.

    > The generated model (along with a graph of training statistics) will be saved at _models/`MODELTYPE`_

    Example:
    ```bash
    python scripts/generate_model.py -m robertav4
    ```

    > If you do not want to train a new model, a `robertav4` model pretrained by us can be found [here](https://drive.google.com/drive/folders/17fma-EUg05jgef1wKoX5o04tjGxdRKb6?usp=sharing)

2. Infer the output values for the [provided test dataset](https://storage.googleapis.com/challenges_events/03_2023/Oracle%202nd%20Reto/Data/test.csv) using `process_data.py` script. You may specyfy the following parameters:

    `-m MODELTYPE`, to declare the base classification model type of the model to infer from, as defined in the previous section.

    `-i MODELPATH`, to set the model file to use for inference, either:
    
      - A relative or absolute path to a `.pt` file, corresponding to a model of type MODELTYPE.
      - `all`, to perform inference for all models at _models/`MODELTYPE`_.
      - Unspecified, to perform inference for the model with higher accuracy at _models/`MODELTYPE`_.

    `--help`, to show the help message for the script.

    > The generated output _json_ will be saved at _results/`MODELTYPE`/[MODEL-UUID].json_

    Example:
    ```bash
    python scripts/process_data.py -m robertav4
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributing

This project is being developed during the course of a competition, so PRs from people outside the competition will **not** be **allowed**. Feel free to fork this repo and follow up the development as you see fit.

Don't forget to give the project a star!

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

Iago Barreiro Río - i.barreiro.rio@gmail.com

Santiago Pérez Acuña - santiago@perezacuna.com

Victor Figueroa Maceira - victorfigma@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>
