num_epochs=100
num_paraphrased=4

# Define the SIF and the script
SIF_FILE="/cbica/home/leejose/finetune_pytorch_2_4.sif" # Or the full path
# Execute the job with the runtime fix
apptainer exec \
    --nv \
    ${SIF_FILE} \
    python finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-13B \
    --device_batch_size 1 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --gradient_checkpointing \
    --offload_to_cpu \
    --compile \
    --full_finetuning

num_epochs=100
num_paraphrased=19

# Execute the job with the runtime fix
apptainer exec \
    --nv \
    ${SIF_FILE} \
    python finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-13B \
    --device_batch_size 1 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --gradient_checkpointing \
    --offload_to_cpu \
    --compile \
    --full_finetuning