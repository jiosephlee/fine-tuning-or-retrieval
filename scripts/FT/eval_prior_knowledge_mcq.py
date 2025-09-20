import os
import json
import sys
import re
import threading
import random
import openai
import asyncio
import pandas as pd

from tqdm.asyncio import tqdm_asyncio
from glob import glob
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

PAPERS = ['1_58', 'BOFT', 'DPO', 'GRPO', 'OFT', 'QLoRA']
MODELS = ['jiosephlee/6_papers_50_epochs', 'allenai/OLMo-2-0425-1B', 'allenai/OLMo-2-0425-1B-Instruct', 'allenai/OLMo-2-1124-7B', 'allenai/OLMo-2-1124-7B-Instruct']
OUTPUT_DIR = '../../results/prior_knowledge/mcq'
mcqa_decoding_params = GuidedDecodingParams(choice=['A', 'B', 'C', 'D'])
mcqa_sampling_params = SamplingParams(guided_decoding=mcqa_decoding_params, temperature=0)

def load_questions():
    mcqs = dict()
    for paper in PAPERS:
        for fp in glob(f'../../data/arxiv/prior_knowledge_mcq/{paper}/*'):
            chapter_name = os.path.basename(fp).split('.')[0]
        with open(fp) as f:
            chapter_qs = json.loads(f.read())
        for i,q in enumerate(chapter_qs['questions']):
            mcqs[f'{paper}/{chapter_name}/q{i}'] = q
    return mcqs

def split_data_for_icl(mcqs, n_icl, SEED=301):
    random.seed(SEED)
    shuffled_mcqs = random.sample(mcqs.items(), k=len(mcqs))
    icl_shots = shuffled_mcqs[:n_icl]
    test_set = shuffled_mcqs[n_icl:]

    return icl_shots, test_set

def qa_template(question, choice_A, choice_B, choice_C, choice_D, correct_answer, test_mode=False):
    return f"""Question: {question}
A) {choice_A}
B) {choice_B}
C) {choice_C}
D) {choice_D}
Answer: {'' if test_mode else correct_answer}"""

def prompt_with_icl(example, icl_shots):
    qid, question = example
    
    prompt = ''
    for _, q in icl_shots:
        prompt += qa_template(**q) + '\n\n'
    gt_answer = question['correct_answer']
    prompt += qa_template(**question, test_mode=True)

    return qid, prompt, gt_answer

def run_mcqs(test_set, llm, icl_shots):
    qids, prompts, gt_answers = zip(*[prompt_with_icl(example, icl_shots) for example in test_set])
    completions = llm.generate(prompts)
    results = [c.outputs[0].text for c in completions]

    return qids, results, gt_answers

if __name__ == '__main__':
    mcqs = load_questions()
    for model in MODELS:
        llm = LLM(model, tensor_parallel_size=1)

        results = []
    for n_icl in [0, 5, 10]:
        icl_shots, test_set = split_data_for_icl(mcqs, n_icl) 
        qids, results, gt_answers = run_mcqs(test_set, llm, icl_shots)
        results += [(qid, r, gt, n_icl) for qid, r, gt in zip(qids, results, gt_answers)]
    
    df = pd.DataFrame(data=results, columns=['question_id', 'prediction', 'ground_truth', 'n_icl'])
    model_name = model.split('/')[1]
    df.to_csv(OUTPUT_DIR + f'/{model_name}.csv')
        
