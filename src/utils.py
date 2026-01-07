import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)

def load_models(draft, target, device):
    draft_model = AutoModelForCausalLM.from_pretrained(draft, torch_dtype=torch.float16).to(device)
    target_model = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.float16).to(device)

    draft_model.eval()
    target_model.eval()

    return draft_model, target_model

def measure_time(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    end = time.perf_counter()

    return out, (end-start)

def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)