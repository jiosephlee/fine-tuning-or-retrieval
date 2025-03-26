import json
import pandas as pd
import datasets

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
         'training_set_filepath': "pqa_artificial",
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
        data = datasets.load_dataset(_DATASETS[dataset]['dataset_name'], path)
        if start_index is not None and end_index is not None:
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