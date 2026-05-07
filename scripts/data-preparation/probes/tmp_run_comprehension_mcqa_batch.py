import importlib.util
from pathlib import Path


DOMAINS = [
    'DPO',
    'BOFT',
    '1_58',
    'OFT',
    'QLoRA',
    'GRPO',
    'ByteLatent',
    'FeatLLM',
    'GSPO',
    'LongRoPE',
    'fa3',
    'xLSTM',
    'Santos_v_Kimmel',
    'Multiphasic_anaphylaxis_in_the_emergency_and_intensive_care',
]


def load_pipeline_module(project_root: Path):
    script_path = project_root / 'scripts' / 'data-preparation' / 'probes' / 'pipeline_generate_comprehension_mcqa.py'
    spec = importlib.util.spec_from_file_location('pipeline_generate_comprehension_mcqa', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    project_root = Path(__file__).resolve().parents[3]
    module = load_pipeline_module(project_root)

    for domain in DOMAINS:
        module.process_domain(domain)


if __name__ == '__main__':
    main()
