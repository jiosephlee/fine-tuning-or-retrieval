num_epochs=100
num_paraphrased=0
nohup python finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 4 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --context_length_for_lima 2560 \
    --full_finetuning > output_1.log 
# This may take around 3 hours
