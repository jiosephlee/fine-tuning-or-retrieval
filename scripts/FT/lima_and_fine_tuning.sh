num_epochs=75
num_paraphrased=0
echo "This is a test run to see if LIMA-based instruction tuning works. Using single arxiv paper but with overlapping sections."\

nohup python finetuning_knowledge_v7.py \
    --custom_suffix "no_overlapping_abalation" \
    --num_train_epochs $num_epochs \
    --learning_rate 2e-5 \
    --num_paraphrased_texts $num_paraphrased \
    --full_finetuning > output_1.log 2>&1 &

        # --lima_afterwards \
