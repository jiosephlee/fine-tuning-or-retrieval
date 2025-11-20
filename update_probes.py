import csv
import re

file_path = 'data/probes/inference/DPO/probes_v7.csv'

def get_new_inference_type(row):
    question = row['question']
    current_type = row['inference_type']
    
    # Check for Counterfactual
    # Look for "if", "would", "expect" in a hypothetical context
    lower_q = question.lower()
    
    is_counterfactual = False
    if 'if ' in lower_q or 'would ' in lower_q or 'suppose ' in lower_q:
        if 'would you therefore expect' in lower_q:
            is_counterfactual = True
        elif 'if ' in lower_q and ('would' in lower_q or 'expect' in lower_q or 'converge' in lower_q or 'fail' in lower_q):
            is_counterfactual = True
        elif 'if ' in lower_q and 'then ' in lower_q:
             is_counterfactual = True
        # Check specific cases from file content I've seen
        if 'if the learned reward model were perfectly accurate' in lower_q:
             is_counterfactual = True
        if 'if human preferences depended on' in lower_q:
             is_counterfactual = True
        if 'if the optimal policy' in lower_q and 'equals' in lower_q: # Math implication
             is_counterfactual = True

    if is_counterfactual:
        return 'Counterfactual Scenarios'
        
    # Default to Conceptual Synthesis as per user instruction for "implied in text"
    # The user said "if answer is somewhat implied in the text sentence... it should be Conceptual Synthesis"
    # This seems to cover the standard inference cases.
    return 'Conceptual Synthesis'

rows = []
with open(file_path, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        new_type = get_new_inference_type(row)
        # Preserve Analogous if it's clearly an analogy? 
        # User didn't explicitly say to remove it, but said "if implied... Conceptual Synthesis".
        # "is most analogous to" -> Analogous Understanding.
        # I'll stick to the user's requested 2 categories mostly, but maybe keep Analogous if it's distinct?
        # The user gave examples for "Conceptual Synthesis" and "Counterfactual".
        # I'll stick to those two to be safe, or maybe map Analogous -> Conceptual Synthesis (Analogy)
        # But user said "it should be Conceptaul Synthesis" (exact phrase).
        
        # Let's just use 'Conceptual Synthesis' and 'Counterfactual Scenarios'.
        
        # Refine Counterfactual check:
        # "would you therefore expect gradient variance... to typically be"
        # "If the learned reward model were perfectly accurate... toward which distribution would..."
        # "If human preferences depended on... what key simplification... would fail?"
        # "When no SFT model is available..." -> This is a "When X, do Y" (Procedural/Factual from text). Not strictly counterfactual/hypothetical in the "what if" sense, but "what is the recommendation". Text says "we initialize...". This is Factual/Conceptual.
        # "If the optimal policy equals the reference policy... what form must the reward take?" -> This is Math derivation. "If X then Y". User said "even if it involves an equation, it should be counterfactual". So yes.
        
        rows.append({**row, 'inference_type': new_type})

# Print some examples to verify
for i, row in enumerate(rows):
    if i < 5 or row['inference_type'] == 'Counterfactual Scenarios':
        print(f"Q: {row['question'][:50]}... -> {row['inference_type']}")

# Write back
with open(file_path, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

