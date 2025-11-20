import csv

file_path = 'data/probes/inference/GRPO/probes_v7.csv'

def get_new_inference_type(row):
    question = row['question']
    
    lower_q = question.lower()
    
    is_counterfactual = False
    if 'if ' in lower_q or 'would ' in lower_q or 'suppose ' in lower_q:
        if 'would you therefore expect' in lower_q:
            is_counterfactual = True
        elif 'if ' in lower_q and ('would' in lower_q or 'expect' in lower_q or 'converge' in lower_q or 'fail' in lower_q or 'happen' in lower_q or 'change' in lower_q):
            is_counterfactual = True
        elif 'if ' in lower_q and 'then ' in lower_q:
             is_counterfactual = True
        # Check specific cases
        if 'if grpo were modified' in lower_q:
             is_counterfactual = True
        if 'if deepseekmath had been initialized' in lower_q:
             is_counterfactual = True
        if 'if the deepseekmath corpus had been english-only' in lower_q:
             is_counterfactual = True
        if 'if grpo is run with group size' in lower_q:
             is_counterfactual = True
        if 'if the team had initialized' in lower_q: # From previous read
             is_counterfactual = True
        if 'if the corpus were english-only' in lower_q:
             is_counterfactual = True
        if 'if all group rewards are identical' in lower_q:
             is_counterfactual = True
             
    if is_counterfactual:
        return 'Counterfactual Scenarios'
        
    return 'Conceptual Synthesis'

rows = []
with open(file_path, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        new_type = get_new_inference_type(row)
        rows.append({**row, 'inference_type': new_type})

# Print some examples to verify
print("Processing GRPO file...")
for i, row in enumerate(rows):
    if row['inference_type'] == 'Counterfactual Scenarios':
        print(f"Q: {row['question'][:50]}... -> {row['inference_type']}")

# Write back
with open(file_path, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

