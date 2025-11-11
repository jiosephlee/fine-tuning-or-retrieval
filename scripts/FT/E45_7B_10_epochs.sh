num_epochs=10
num_paraphrased=0
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
    --full_finetuning > output_45_a.log   

num_epochs=10
num_paraphrased=9
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
    --full_finetuning > output_45_b.log   

# 2 hours
