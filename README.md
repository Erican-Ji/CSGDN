# CSGDN

<>

## Frameworks

!(Frameworks)[]

## Abstract

## Dataset

There is no dataset provided in the code, if you want to use it, please download and calculate it yourself and put it into `data`, e.g. `data/TWAS.xlsx`

|data|data_name|
|:-:|:-:|
|Ghirsutum|<https://Ghirsutum.hzau.edu.cn/EN/Download.htm>|
|Brassica napus|/|
|Wheat|/|

For the node features (i.e. gene similarity matrix), please calculate them yourself and put them into `data/data_name/ori_sim.txt`, e.g. `data/cotton/ori_sim.txt`

Below is a diagram showing the format of our data:

![dataset_format](images/dataset_format.png)

## Usage

1. Generate data for the experiment

    When the dataset is ready, run

    ``bash
    python data_generator.py --dataset cotton
    ``

    The corresponding training, validation and test data of cotton will be obtained, and the corresponding node features and the graph after Diffusion will be generated automatically. If you want to change the threshold for the Diffusion operation, add it yourself to the function called in the last line of data_generator.py.

2. Training

    Run

    ```bash
    python train.py --dataset cotton
    ```

    to get the final result

We've set up a series of hyperparameters that can be tweaked

|hyperparameters|usage|default|
|:-:|:-:|:-:|
|alpha|--alpha|0.8|
|beta|--beta|0.01|
|feature_dim|--feature_dim|64|
|mask_ratio|--mask_ratio|0.4|
|predictor|--predictor|2|
|tau|--tau|0.05|