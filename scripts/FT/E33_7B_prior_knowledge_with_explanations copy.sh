num_epochs=100
num_paraphrased=9
python finetuning_knowledge_v8.py \
    --model_id jiosephlee/prior_6_papers_25_epochs_7B \
    --device_batch_size 4 \
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
    --context_length_for_lima 2560 \
    --full_finetuning > output_8.log 2>&1

    
# 6-8 hours