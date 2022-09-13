
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.models.transformer_rep import Splade

# set the dir for trained weights

##### v2
# model_type_or_dir = "weights/splade_max"
# model_type_or_dir = "weights/distilsplade_max"

### v2bis, directly download from Hugging Face
# model_type_or_dir = "naver/splade-cocondenser-selfdistil"
model_type_or_dir = "naver/splade-cocondenser-ensembledistil"

# loading model and tokenizer

model = Splade(model_type_or_dir, agg="max")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

doc = "Glass and Thermal Stress. Thermal Stress is created when one area of a glass pane gets hotter than an adjacent area. If the stress is too gre"

# now compute the document representation
with torch.no_grad():
    doc_rep = model(d_kwargs=tokenizer(doc, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

# get the number of non-zero dimensions in the rep:
col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
print("number of actual dimensions: ", len(col))

# now let's inspect the bow representation:
weights = doc_rep[col].cpu().tolist()
d = {k: v for k, v in zip(col, weights)}
sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
bow_rep = []
for k, v in sorted_d.items():
    bow_rep.append((reverse_voc[k], round(v, 2)))
print("SPLADE BOW rep:\n", bow_rep)