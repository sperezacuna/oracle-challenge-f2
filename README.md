## About The Project

This is a review classifier based on sentiment analysis built under the specification for the second challenge of the contest [Reto Enseña Oracle](https://nuwe.io/dev/competitions/reto-ensena-oracle-espana/sentimiento-y-deploymentML)

The main functionality of the project can be summarized as follows:

* Generate a model capable of identifying the positive or negative sentiment of a review.
* Fine tune the model using the 
  * Segregate a portion of the dataset for validation purposes and hence avoiding overfitting.
  * Data augmentation using transforms.
* Different implementations of the Roberta and BERT sentiment classifier are available for process the data.
* Inference and result saving for test dataset as _json_ file.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built Using

Base technologies:

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)


Additional dependencies:

* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Transformers](https://huggingface.co/docs/transformers/index)


<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

Given that [python3](https://www.python.org/downloads/) and [pip](https://pypi.org/project/pip/) are installed and correctly configured in the system, and that you have a [CUDA-capable hardware](https://developer.nvidia.com/cuda-gpus) installed, you may follow these steps.

### Prerequisites

* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 11.0 or above is correctly installed.
* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) version 7 or above is correctly installed.
* You will require at least 12GB of VRAM to load the robertav3 and robertav4 models onto the GPU, and 6GB for the rest

### Installation

1. Clone this repository locally.

```bash
git clone git@github.com:sperezacuna/oracle-challenge-f2.git
```
2. Create python [virtual environment](https://docs.python.org/3/library/venv.html) and activate it (**recommended**)

```bash
python -m venv env
source env/bin/activate 
```

3. Install all required dependencies.

```bash
pip install -r requirements.txt
```

### Execution

To train a new model based on the [train dataset](https://storage.googleapis.com/challenges_events/03_2023/Oracle%202nd%20Reto/Data/train.csv) and the [test dataset](https://storage.googleapis.com/challenges_events/03_2023/Oracle%202nd%20Reto/Data/test.csv). 
Then generate the validation dataset using the 10% os the train datset.

1. Generate de model.

```bash
python generate_model.py [-m MODELTYPE]
```

2. Process the data.

```bash
python process_data.py [-m MODELTYPE] [-i MODELFILE]
```

3. You will use the test dataset to evaluate your model's F1-score on the [Nuwe](https://nuwe.io/dev/competitions/reto-ensena-oracle-espana/sentimiento-y-deploymentML).

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
