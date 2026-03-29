import os
import pandas as pd

def extract_questions():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    domains = ['DPO', 'BOFT', '1_58', 'OFT', 'QLoRA', 'GRPO']
    
    for domain in domains:
        input_path = os.path.join(base_dir, f'data/probes/inference/{domain}/probes_v7.csv')
        output_path = os.path.join(base_dir, f'data/probes/inference/{domain}/questions_v7.txt')
        
        if os.path.exists(input_path):
            try:
                df = pd.read_csv(input_path)
                if 'question' in df.columns:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for question in df['question']:
                            f.write(f"{question}\n")
                    print(f"Extracted {len(df)} questions to {output_path}")
                else:
                    print(f"Warning: 'question' column not found in {input_path}")
            except Exception as e:
                print(f"Error processing {domain}: {e}")
        else:
            print(f"Input file not found: {input_path}")

if __name__ == "__main__":
    extract_questions()
