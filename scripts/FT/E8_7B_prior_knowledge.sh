num_epochs=50
num_paraphrased=0
python finetuning_prior_knowledge_v1.py \
    --model_id allenai/OLMo-2-1124-7B \
    --device_batch_size 4 \
    --effective_batch_size_for_cpt 256 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 4e-5 \
    --lima_afterwards \
    --push_to_hub_cpt_id prior_6_papers_50_epochs_7B \
    --push_to_hub_lima_id prior_6_papers_50_epochs_7B_with_lima \
    --full_finetuning > output_1.log

# Time: 4 hours