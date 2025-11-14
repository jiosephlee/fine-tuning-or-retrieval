num_epochs=100
num_paraphrased=0
SIF_FILE="/cbica/home/leejose/pytorch_24.05-py3.sif" # Or the full path
# Execute the job with the runtime fix
apptainer exec \
    --nv \
    ${SIF_FILE} \
    python -s finetuning_knowledge_v8.py \
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
    --gradient_checkpointing \
    --constant_lr \
    --full_finetuning > output_55.log   


num_epochs=100
num_paraphrased=9
apptainer exec \
    --nv \
    ${SIF_FILE} \
    python -s finetuning_knowledge_v8.py \
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
    --gradient_checkpointing \
    --constant_lr \
    --full_finetuning > output_55.log   

num_epochs=100
num_paraphrased=9
apptainer exec \
    --nv \
    ${SIF_FILE} \
    python -s finetuning_knowledge_v8.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 4 \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --with_explanations \
    --constant_lr \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --gradient_checkpointing \
    --full_finetuning > output_55_c.log   


