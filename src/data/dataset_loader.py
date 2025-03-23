import warnings
from datasets import load_dataset

warnings.simplefilter("ignore")
dataset = load_dataset("conll2003", trust_remote_code=True)
print(dataset)