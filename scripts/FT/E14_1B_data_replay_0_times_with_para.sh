num_epochs=100
num_paraphrased=9
python finetuning_knowledge_v8.py \
    --device_batch_size 16 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning \
    --context_length_for_lima 2560 > output_1.log 

# This may takes around 4 hours to run on 3090 GPU
