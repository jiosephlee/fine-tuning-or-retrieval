num_epochs=80
num_paraphrased=0
echo hey!
nohup python finetuning_prior_knowledge_v1.py \
    --effective_batch_size_for_cpt 16 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_1.log 2>&1 &

        # --separate_batches_with_pretraining 1 \

