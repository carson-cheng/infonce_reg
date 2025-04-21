# Mutual Information Regularization with InfoNCE Loss in Supervised Classification Problems

Code for experiments regarding the use of InfoNCE loss and feature-label mutual information regularization by fine-tuning different base models on different image classification datasets

### Prerequisites

- Python 3.x

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/carson-cheng/infonce_reg.git
   cd infonce_reg
   ```
2. Install the required dependencies:
   ```
   pip install requirements.txt
   ```

### Running the Experiments

The code accepts inputs in the following format:
```
python3 main.py [base_model_name] [dataset_name]
```
where [base_model_name] can be selected from ['resnet18', 'wide_resnet101_2', 'vit_b_16'] and [dataset_name] can be selected from ['cifar10', 'cifar100', 'cars', 'dogs', 'flowers']