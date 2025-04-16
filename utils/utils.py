import json
import pandas as pd
import datasets
import os
import time
from openai import OpenAI
from utils.keys import OPENAI_API_KEY

def save_predictions():
    pass

def evaluate_metrics():
    pass

#########################
## For Dataset Processing
#########################

_PROJECT_PATH = "/Users/jlee0/Desktop/research/fine-tuning-or-retrieval"

_DATASETS = {
    "MedQA": 
        {'test_set_filepath': f"{_PROJECT_PATH}/data/MedQA/questions/US/4_options/phrases_no_exclude_test.jsonl",
        'format': 'jsonl',
        },
    "BioASQ": 
        {'test_set_filepath': "./data/mimic-iv-public/triage_counterfactual.csv",
        'format': 'csv',
        'target': 'acuity',
        'training_set_filepath':'./data/mimic-iv-public/triage_public.csv',
        },
    "PubMedQA": 
        {'dataset_name': "qiaojin/PubMedQA",
         'train_set_filepath': "pqa_artificial",
         'test_set_filepath': "pqa_labeled",
        'format': 'huggingface',
        },
    }

def save_metrics(metrics,  filename):
    output_file = f"{_PROJECT_PATH}/results/metrics/{filename}_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
             
def load_dataset(dataset, start_index=None, end_index=None, split='test'):
    if dataset not in _DATASETS:
        raise ValueError("Dataset not found in _DATASETS.")
    if _DATASETS[dataset]['format'] == 'huggingface':
        path = _DATASETS[dataset][f'{split}_set_filepath'] 
        data = datasets.load_dataset(_DATASETS[dataset]['dataset_name'], path)['train']
        print(data)
        if start_index is not None and end_index is not None:
            # Using slicing instead of select method
            data = data.select(range(start_index, end_index))
    else:
        filepath = _DATASETS[dataset]['test_set_filepath']
        format = _DATASETS[dataset]['format']
        if format == 'jsonl':
            data = load_jsonl(filepath, start_index, end_index)
        elif format == 'csv':
            data = pd.read_csv(filepath).loc[start_index:end_index]
        else:
            raise ValueError(f"Unsupported format: {format}")
    return data
    
def load_jsonl(filepath, start_index, end_index):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for i, line in enumerate(f) if i <= end_index and i >= start_index]
    return data

def load_predictions(filename, format='txt', save_path=f"{_PROJECT_PATH}/results/predictions"):
    if format == 'csv':
        filename = f"{save_path}/{filename}.csv"
        predictions = pd.read_csv(filename)
    else: 
        filename = f"{save_path}/{filename}.txt"
        with open(filename, 'r') as f:
            predictions = [json.loads(line.strip()) for line in f]
    return predictions

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()


def query_llm(prompt, max_tokens=1000, temperature=0, top_p = 0, max_try_num=3, model="gpt-4o-mini", debug=False, return_json=False, logprobs=False, system_prompt_included=False, response_format=None):
    if debug:
        print(prompt)
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model or 'o3' in model:
                messages = []
                if system_prompt_included and isinstance(prompt, dict) and "system" in prompt:
                    messages.append({"role": "system", "content": prompt["system"]})
                    if "user" in prompt:
                        messages.append({"role": "user", "content": prompt["user"]})
                else:
                    messages.append({"role": "user", "content": prompt})
                
                if model == "o3-mini":
                    api_params = {
                        "model": model,
                        "messages": messages,
                        'reasoning_effort': 'low',
                    }
                else :
                    api_params = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "seed": 0
                    }
                
                if response_format is not None:
                    api_params["response_format"] = response_format
                
                elif return_json:
                    api_params["response_format"] = {"type": "json_object"}
                
                if logprobs:
                    api_params["logprobs"] = logprobs
                    api_params["top_logprobs"] = 3
                
                if response_format is not None:
                    completion = client.beta.chat.completions.parse(**api_params)
                else:
                    completion = client.chat.completions.create(**api_params)
                
                if response_format is not None and hasattr(completion.choices[0].message, 'parsed'):
                    response = completion.choices[0].message.parsed
                else: 
                    response = completion.choices[0].message.content.strip()
                
            if debug:
                print(response)
            if logprobs:
                return response, completion.choices[0].logprobs
            return response
        except Exception as e:
            if 'gpt' in model:
                print(f"Error making OpenAI API call: {e}")
            else: 
                print(f"Error making API call: {e}")
            curr_try_num += 1
            if curr_try_num >= max_try_num:
                return (-1)
            time.sleep(10)