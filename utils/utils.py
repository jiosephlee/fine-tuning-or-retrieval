import json
import pandas as pd
import datasets
import os
import time
from openai import OpenAI
from utils.keys import OPENAI_API_KEY, DATABRICKS_TOKEN


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

client_safe = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-4750903324350629.9.azuredatabricks.net/serving-endpoints"
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()


def query_llm(prompt, max_tokens=1000, temperature=0, top_p=0, max_try_num=10, model="gpt-4o-mini", debug=False, return_json=False, json_schema=None, logprobs=False, system_prompt_included=True, is_hippa=False):
    if debug:
        if system_prompt_included:
            print(f"System prompt: {prompt['system']}")
            print(f"User prompt: {prompt['user']}")
        else:
            print(prompt)
        print(f"Model: {model}")
    if is_hippa and ('gpt' not in model and 'o3' not in model):
        raise ValueError("HIPPA compliance requires GPT models")
    curr_try_num = 0
    while curr_try_num < max_try_num:
        try:
            if 'gpt' in model or 'o3' in model or 'o1' in model or 'o4' in model:
                response = query_gpt(prompt, model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, return_json=return_json, json_schema=json_schema, logprobs=logprobs, system_prompt_included=system_prompt_included, is_hippa=is_hippa, debug=debug)
                if logprobs:
                    return response.choices[0].message.content.strip(), response.choices[0].logprobs
            return response
        except Exception as e:
            raise e
    return None

def query_gpt(prompt: str | dict, model: str = 'gpt-4.1-mini', max_tokens: int = 4000, temperature: float = 0, top_p: float = 0, logprobs: bool = False, return_json: bool = False, json_schema = None, system_prompt_included: bool = False, reasoning_effort = 'high', is_hippa: bool = False, debug: bool = False):
    """OpenAI API wrapper; For HIPPA compliance, use client_safe e.g. model='openai-gpt-4o-high-quota-chat'"""
    temp_client = client_safe if is_hippa else client

    if system_prompt_included:
        # Format chat prompt with system and user messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    if 'gpt-5' in model:
        api_params = {
            "model": model,
            "messages": messages,
            "reasoning_effort": reasoning_effort,
        }
    elif 'o3' in model or 'o1' in model or 'o4' in model:
        api_params = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "messages": messages,
        }
    else:
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": 0
        }
    if logprobs:
        api_params["logprobs"] = logprobs
        api_params["top_logprobs"] = 3

    if return_json:
        if json_schema is None:
            api_params["response_format"] = {"type": "json_object"}
            completion = temp_client.chat.completions.create(**api_params)
            response = completion.choices[0].message.content.strip()
        else:
            api_params["response_format"] = json_schema
            completion = client.beta.chat.completions.parse(**api_params)
            response = completion.choices[0].message.parsed
    else: 
        completion = temp_client.chat.completions.create(**api_params)
        response = completion.choices[0].message.content.strip()
    if debug:
        print(f"Response: {response}")
    if logprobs:
        return response, completion.choices[0].logprobs
    else:
        return response