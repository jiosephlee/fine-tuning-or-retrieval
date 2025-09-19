num_epochs=100
num_paraphrased=9
nohup python finetuning_knowledge_v8.py \
    --override_domains DPO 1_58 GRPO BOFT OFT QLoRA \
    --with_explanations \
    --effective_batch_size_for_cpt 32 \
    --separate_batches_with_pretraining 1 \
    --pretraining_data_type dclm \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_1.log 2>&1 &

        # --separate_batches_with_pretraining 1 \

