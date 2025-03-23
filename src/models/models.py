from transformers import AutoModelForTokenClassification
from src.data.processiing import label_list, label_to_id, id_to_label

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-large-finetuned-conll03-english",
    num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id,
    ignore_mismatched_sizes=True
)