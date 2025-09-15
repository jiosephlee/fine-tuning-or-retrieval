num_epochs=100
num_paraphrased=0
echo "This is a test run to see if LIMA-based instruction tuning works. Using single arxiv paper but with overlapping sections."\

nohup python finetuning_knowledge_v8.py \
    --test_script \
    --override_domains DPO \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --overlap_sections \
    --overlap_ratio 1_4 \
    --lima_afterwards \
    --full_finetuning > output_1.log 2>&1 &

    #--custom_suffix "no_overlapping_abalation_no_title" \


    
