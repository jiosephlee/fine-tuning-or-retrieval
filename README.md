# fine-tuning-or-retrieval

To run an experiment, first you will need to install these libraries (some bug fixes have not been pushed to the wheels available on pip and must be compiled from soruce)

```pip install flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/trl
pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
git clone https://github.com/jiosephlee/transformers
cd transformers && pip install .[torch]
```

Then, go to the script `scripts/FT/lima_and_fine_tuning.sh`

which looks like this

```python finetuning_knowledge_v7.py \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --do_eval \
    --lima_afterwards \
    --full_finetuning > output_1.log
```


    
