num_epochs=80
num_paraphrased=0
echo hey!
sleep 4400
python finetuning_knowledge_v8.py \
    --override_domains DPO 1_58 GRPO \
    --effective_batch_size_for_cpt 16 \
    --fill_batches_with_pretraining \
    --separate_batches_with_pretraining 1 \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_1.log 2>&1

    #--custom_suffix "no_overlapping_abalation_no_title" \

python finetuning_knowledge_v8.py \
    --override_domains DPO 1_58 GRPO \
    --effective_batch_size_for_cpt 16 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_2.log 2>&1

num_paraphrased=9

python finetuning_knowledge_v8.py \
    --override_domains DPO 1_58 GRPO \
    --effective_batch_size_for_cpt 16 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_3.log 2>&1

python finetuning_knowledge_v8.py \
    --override_domains DPO 1_58 GRPO \
    --effective_batch_size_for_cpt 16 \
    --fill_batches_with_pretraining \
    --separate_batches_with_pretraining 1 \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_4.log 2>&1

