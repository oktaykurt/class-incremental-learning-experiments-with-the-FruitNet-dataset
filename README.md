# Class-Incremental Learning Experiments with FruitNet

## Project Overview

This project investigates Class-Incremental Learning (Class-IL) techniques to address catastrophic forgetting in neural networks when learning new classes sequentially. Using the FruitNet dataset, this work empirically compares several continual learning strategies, analyzing their ability to balance learning new information (plasticity) while retaining old knowledge (stability). The methods compared include Fine-Tuning, Joint training (as an upper-bound reference), a replay-based approach, Elastic Weight Consolidation (EWC), and Learning without Forgetting (LwF). The experiments are implemented using PyTorch with a ResNet-18 backbone.

![restnet18](https://github.com/user-attachments/assets/93581376-0dd3-4651-832f-77ad93134a8e)


## Project Report

For a detailed explanation of the methodology, experiments, results, and analysis, please see the full report:
[View Full Project Report (PDF)](https://github.com/oktaykurt/class-incremental-learning-experiments-with-the-FruitNet-dataset/blob/main/Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset-Report.pdf)

## Dataset: FruitNet

![samples](https://github.com/user-attachments/assets/72afede1-9ba2-4b1a-aa15-5c99271027e3)

* **Source:** FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)
    * Available at: [Mendeley Data](https://data.mendeley.com/datasets/b6fftwbr2v/1) and [Kaggle](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality)
* **Content:** The dataset contains images of 6 types of Indian fruits (this project uses 5: Apple, Banana, Guava, Lime, Orange) categorized by quality (Good, Bad).
* **Task Setup:** The project defines a Class-Incremental Learning scenario with 5 sequential tasks. Each task introduces two new classes corresponding to the Good and Bad quality images of one fruit type (e.g., Task 0: Apple_Good, Apple_Bad; Task 1: Banana_Good, Banana_Bad, etc.).
* **Preprocessing:** Images are resized to 192x192 pixels and normalized using standard ImageNet statistics. The dataset is split 80% for training and 20% for testing.

## Data Citation

If you use the FruitNet dataset, please cite the original source:

> MESHRAM, Vishal; PATIL, Kailas (2021), “FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)”, Mendeley Data, V1, doi: 10.17632/b6fftwbr2v.1

## Methods Compared

This project implements and compares the following Class-Incremental Learning strategies:

1.  **Fine-Tuning:** A baseline approach where the model is trained sequentially only on the data for the current task. Prone to catastrophic forgetting.
2.  **Joint Training:** A non-continual upper bound where the model is retrained from scratch on the accumulated data from all tasks seen so far at each step.
3.  **Replay (referred to as iCaRL in code/report):** A replay-based method that stores a small number of exemplar images from previous tasks and includes them in the training data for subsequent tasks. (See Note below regarding implementation).
4.  **EWC (Elastic Weight Consolidation):** A regularization-based method that penalizes changes to weights deemed important for previous tasks, calculated using the Fisher Information Matrix.
5.  **LwF (Learning without Forgetting):** A distillation-based method where the model trained on a new task is encouraged to maintain similar output probabilities for previous task classes as the model state before learning the new task.

![perf_current](https://github.com/user-attachments/assets/93659854-bbf0-4b90-9475-983cabc16443)
![perf_past](https://github.com/user-attachments/assets/df117f7e-0af9-4894-a81a-e9d05e54fb4c)

## Setup & Requirements

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy Pillow matplotlib seaborn
    ```
3.  **Prepare Data:**
    * Download the FruitNet dataset from [Mendeley Data](https://data.mendeley.com/datasets/b6fftwbr2v/1) or [Kaggle](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality).
    * Organize the data according to the structure expected by the notebook (e.g., separate folders for 'Good Quality_Fruits' and 'Bad Quality_Fruits', each containing subfolders for fruit types).
    * Update the data paths (`root_good`, `root_bad`) in the notebook (`Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset.ipynb`) to point to your dataset location.

## Usage

1.  Ensure your data is set up correctly and the paths are updated in the notebook.
2.  Run the `Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset.ipynb` notebook using Jupyter:
    ```bash
    jupyter notebook Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset.ipynb
    ```
3.  Execute the cells sequentially. The notebook will:
    * Load and prepare the data.
    * Define the model and utility functions.
    * Run experiments for each Class-IL method (Fine-Tuning, Joint, Replay/iCaRL, EWC, LwF).
    * Perform hyperparameter grid searches for Replay/iCaRL, EWC, and LwF.
    * Generate comparison plots (R-matrices, confusion matrices, performance curves, resource usage, plasticity vs. stability).

## Results

The notebook generates detailed results and visualizations comparing the different Class-IL methods based on accuracy on current and past tasks, forgetting metrics, model parameter growth, and training time. Key plots include the R-matrix for each method, confusion matrices, performance over time, and a plasticity vs. stability comparison plot. Please refer to the notebook and the [Project Report](https://github.com/oktaykurt/class-incremental-learning-experiments-with-the-FruitNet-dataset/blob/main/Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset-Report.pdf) for detailed findings.

## Files Included

* `Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset.ipynb`: Jupyter Notebook containing the full code for data loading, preprocessing, model definitions, Class-IL method implementations, experiments, and result visualization.
* `Class_Incremental_Learning_Experiments_with_the_FruitNet_Dataset-Report.pdf`: A detailed report discussing the Class-IL problem, dataset, methodology, experimental results, and conclusions.

## Note on "iCaRL" Implementation

Please note that the method referred to as "iCaRL" in the code and report is a simplified replay-based strategy using a fixed number of randomly selected exemplars per class. It does not implement all components of the original iCaRL paper, such as the Nearest-Mean-of-Exemplars classifier. It serves as a representative example of a basic replay approach in these experiments.
