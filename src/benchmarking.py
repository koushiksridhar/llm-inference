import yaml
import pandas as pd
import torch
from tqdm import tqdm
from decoders import baseline_decode, speculative_decode
from utils import load_models, load_tokenizer, measure_time

def run_benchmark():
    # Load configs
    print('[INFO] Starting Benchmarking...')

    c_models = yaml.safe_load(open("configs/models.yaml"))
    c_benchmarks = yaml.safe_load(open("configs/benchmarks.yaml"))
    print('[INFO] Loaded configs...')


    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'[INFO] Device + {device}')

    tokenizer = load_tokenizer(c_models['tokenizer'])
    print('[INFO] Loaded tokenizer...')

    draft_model, target_model = load_models( c_models['draft_model'], c_models['target_model'], device)
    print('[INFO] Loaded models...')


    records = []

    for prompt in tqdm(c_benchmarks['prompts']):
        print(f'[INFO] Prompt: {prompt}')

        baseline_out, t_base = measure_time(
            baseline_decode, target_model, tokenizer, prompt, c_benchmarks['max_new_tokens']
        )

        print(f'Ran Baseline @ {t_base}.\nOutput: {baseline_out}')


        records.append({
            'prompt': prompt,
            'method': 'baseline',
            'draft_k': 0,
            'time': t_base,
            'output': baseline_out
        })

        for k in c_benchmarks['draft_lengths']:
            spec_out, t_spec = measure_time(
                lambda *args: speculative_decode(*args, return_metrics=True),
                draft_model,
                target_model,
                tokenizer,
                prompt,
                k,
                c_benchmarks['max_new_tokens']
            )

            spec_text, metrics = spec_out
            print(f"Ran Speculative {k} @ {t_spec}.\nOutput: {spec_text}\nToken Acceptance Rate: {metrics['token_acceptance_rate']}\nFull Accept Rate: {metrics['full_accept_rate']}")            
            records.append({
                'prompt': prompt,
                'method': 'speculative',
                'draft_k': k,
                'time': t_spec,
                'output': spec_text,
                'token_acceptance_rate': metrics['token_acceptance_rate'],
                'full_accept_rate': metrics['full_accept_rate']
            })

    df = pd.DataFrame(records)
    df.to_csv('results/benchmarks.csv', index=False)
    print('[INFO] Saved results...')


if __name__ == '__main__':
    run_benchmark()
