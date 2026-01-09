import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

def baseline_decode(model, tokenizer, prompt, max_new_tokens):
    """ Perform baseline decode via greedy"""
    inputs =  tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return tokenizer.decode(output[0], skip_special_tokens=True)

@torch.no_grad()
def speculative_decode(draft_model, target_model,tokenizer, prompt, draft_k, max_new_tokens, return_metrics=False):
    
    device = target_model.device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    total_generated = 0

    # Acceptance Stats Tracking
    proposed_tokens = 0
    accepted_tokens = 0
    speculative_steps = 0
    full_accepts = 0

    while total_generated < max_new_tokens:

        speculative_steps += 1

        # Draft proposes k tokens
        draft_input = generated
        draft_tokens = []

        for _ in range(draft_k):
            logits = draft_model(draft_input).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            draft_tokens.append(next_token)
            draft_input = torch.cat([draft_input, next_token], dim=-1)

        draft_tokens = torch.cat(draft_tokens, dim=-1)
        proposed_tokens += draft_k

        # Target verifies
        target_input = torch.cat([generated, draft_tokens], dim=-1)
        target_logits = target_model(target_input).logits

        start = generated.size(1) - 1
        verify_logits = target_logits[:, start:-1, :]

        # Acceptance check
        accepted = 0
        for i in range(draft_k):
            target_next = torch.argmax(verify_logits[:, i, :], dim=-1)
            if target_next.item() == draft_tokens[:, i].item():
                accepted += 1
            else:
                break

        if accepted > 0:
            generated = torch.cat(
                [generated, draft_tokens[:, :accepted]],
                dim=-1
            )
            accepted_tokens += accepted
            total_generated += accepted

        if accepted == draft_k:
            full_accepts += 1

        # If rejected, continue as if target was used
        if accepted < draft_k:
            logits = target_logits[:, start + accepted, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            total_generated += 1

        if generated.size(1) >= input_ids.size(1) + max_new_tokens:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    if not return_metrics:
        return text

    metrics = {
        "draft_k": draft_k,
        "proposed_tokens": proposed_tokens,
        "accepted_tokens": accepted_tokens,
        "token_acceptance_rate": (
            accepted_tokens / proposed_tokens if proposed_tokens > 0 else 0.0
        ),
        "speculative_steps": speculative_steps,
        "full_accepts": full_accepts,
        "full_accept_rate": (
            full_accepts / speculative_steps if speculative_steps > 0 else 0.0
        ),
    }

    return text, metrics
