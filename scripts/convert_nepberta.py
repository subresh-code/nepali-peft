"""One-off: convert official NepBERTa TF weights to PyTorch.

from_tf=True is broken here (meta-device skeleton, copy is a no-op), so
instantiate a real-tensor BertModel from config and load the TF h5 into
it directly, then verify against the Rajan/nepbertaTorch community port.
"""
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, BertConfig, BertModel
from transformers.modeling_tf_pytorch_utils import (
    load_tf2_checkpoint_in_pytorch_model,
)

OUT = sys.argv[1]

cfg = BertConfig.from_pretrained("NepBERTa/NepBERTa")
model = BertModel(cfg)  # real tensors, not meta
h5 = hf_hub_download("NepBERTa/NepBERTa", "tf_model.h5")
model = load_tf2_checkpoint_in_pytorch_model(model, h5, allow_missing_keys=True)

meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
assert not meta, f"still meta: {meta[:5]}"

# Integrity check 1: not freshly initialized (a real trained LayerNorm
# weight is not all-ones).
ln = model.encoder.layer[0].output.LayerNorm.weight
assert not torch.allclose(ln, torch.ones_like(ln)), "weights look untrained!"

# Integrity check 2: tensor-by-tensor comparison against the community port.
ours = model.state_dict()
port = BertModel.from_pretrained("Rajan/nepbertaTorch").state_dict()
shared = sorted(set(ours) & set(port))
mismatched = [k for k in shared if not torch.equal(ours[k], port[k])]
print(f"compared {len(shared)} shared tensors")
print("only-ours:", sorted(set(ours) - set(port)))
print("only-port:", sorted(set(port) - set(ours)))
print("MISMATCHED:", mismatched if mismatched else "none — port is exact")

model.save_pretrained(OUT)
AutoTokenizer.from_pretrained("NepBERTa/NepBERTa").save_pretrained(OUT)
print("saved to", OUT, "| params:", sum(p.numel() for p in model.parameters()))
