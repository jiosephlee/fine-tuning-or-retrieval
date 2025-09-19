# fine-tuning-or-retrieval

To run an experiment, first you will need to install these libraries (some bug fixes have not been pushed to the wheels available on pip and must be compiled from source)

```
pip install flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/trl
pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
git clone https://github.com/jiosephlee/transformers
cd transformers && pip install .[torch]
```
A incompatibility between TRL and transformers will be reported but that is fine as of September 18th 2025 version of TRL.

You must also create a file called keys.py that contains the API keys necessary for the LLM-as-a-judge evals. Keys_demo.py has been provided for you to edit. Please rename it to keys.py so that it can also be ignored by git.

```
wandb init
huggingface-cli login
```
are also necessary to track experiments and use the LIMA dataset for instruction-tuning.

Then, go to the script `scripts/FT/lima_and_fine_tuning.sh`

which looks like this

```
num_epochs=100
num_paraphrased=10
python finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 2 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_1.log
```


    
