num_epochs=50
num_paraphrased=0
echo hey!
nohup python finetuning_prior_knowledge_v1.py \
    --device_batch_size 4 \
    --effective_batch_size_for_cpt 256 \
    --fill_batches_with_pretraining \
    --num_train_epochs $num_epochs \
    --learning_rate 4e-5 \
    --lima_afterwards \
    --full_finetuning > output_1.log 2>&1 &

        # --separate_batches_with_pretraining 1 \

