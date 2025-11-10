num_epochs=100
num_paraphrased=9
accelerate launch --config_file deepspeed.yaml finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-0325-32B \
    --device_batch_size 1 \
    --with_explanations \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 16 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --gradient_checkpointing \
    --full_finetuning > output_47.log 


# export CUDA_VISIBLE_DEVICES=2,3 to use 2 GPUs for instance.