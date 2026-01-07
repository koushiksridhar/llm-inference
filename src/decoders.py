import torch
from transformers import AutoTokenizer

def baseline_decode(model, tokenizer, prompt, max_new_tokens):
    """ Perform baseline decode via greedy"""
    inputs =  tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return tokenizer.decode(output[0], skip_special_tokens=True)

def speculative_decode(draft_model, target_model, tokenizer, prompt, k, max_new_tokens):
    """Performs speculative decoding with draft_length k"""
    device = draft_model.device
    enc = tokenizer(prompt, return_tensors='pt').to(device)

    # Initialize outputs for both model
    draft_out = draft_model(**enc, use_cache=True)
    target_out = target_model(**enc, use_cache=True)

    draft_cache = draft_out.past_key_values
    target_cache = target_out.past_key_values

    tokens = []

    for step in range(max_new_tokens):
        # Draft model proposes k tokens
        draft_inputs = {"input_ids": None, "past_key_values": draft_cache}
        draft_logits, draft_cache = draft_model(**draft_inputs, return_dict=False, use_cache=True)

        top_tokens = torch.argmax(draft_logits[:, -1], dim=1)
        tokens.extend(top_tokens.tolist())

        # Verify with target model
        verification_inputs = {
            "input_ids": torch.tensor([tokens], device=device),
            "past_key_values": target_cache,
        }
        target_out, new_target_cache = target_model(
            **verification_inputs, return_dict=False, use_cache=True
        )

        # Accept or Reject
        if torch.argmax(target_out[:, -1], dim=1) == top_tokens:
            target_cache = new_target_cache
        else:
            tokens.pop()
            # Fallback to greedy decode
            fallback_input = {
                "input_ids": torch.tensor([tokens], device=device),
                "past_key_values": target_cache,
            }

            greedy_logits, target_cache = target_model(**fallback_input, return_dict=False, use_cache=True)
            greedy_token = torch.argmax(greedy_logits[:, -1], dim=1)

            tokens.append(greedy_token.item())
    
    return tokenizer.decode(tokens, skip_special_tokens=True)