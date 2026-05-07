num_epochs=50
num_paraphrased=0
python finetuning_prior_knowledge_v1.py \
    --device_batch_size 2 \
    --effective_batch_size_for_cpt 256 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 4e-5 \
    --lima_afterwards \
    --push_to_hub_cpt_id prior_6_papers_50_epochs_1B \
    --push_to_hub_lima_id prior_6_papers_50_epochs_1B_with_lima \
    --context_length_for_lima 2560 \
    --full_finetuning > output_1.log 
# Time: 3 hours