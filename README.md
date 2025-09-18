# fine-tuning-or-retrieval

To run an experiment, first you will need to install these libraries (some bug fixes have not been pushed to the wheels available on pip and must be compiled from source)

```pip install flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/trl
pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
git clone https://github.com/jiosephlee/transformers
cd transformers && pip install .[torch]
```

You must also create a file called keys.py that contains the API keys necessary for the LLM-as-a-judge evals. Keys_demo.py has been provided for you to edit. Please rename it to keys.py so that it can also be ignored by git.

Then, go to the script `scripts/FT/lima_and_fine_tuning.sh`

which looks like this

```
num_epochs=100
num_paraphrased=10
python finetuning_knowledge_v7.py \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --do_eval \
    --lima_afterwards \
    --full_finetuning > output.log
```


    
