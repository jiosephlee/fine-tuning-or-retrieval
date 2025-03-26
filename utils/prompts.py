# Function that gives prompt for correspodning 

_PROMPTS = {
    'medqa': {
        'prompt1': ''
    },
    'bioasq': {
        'prompt1': ''
    }
}

BLACKBOX_LLM_PROMPT = """### Instructions
{}

### Patient's Clinical Presentation
{}"""

def get_prompt(dataset_name, prompt_name):
    return _PROMPTS[dataset_name][prompt_name]

def format_prompt_for_finetuning():
    