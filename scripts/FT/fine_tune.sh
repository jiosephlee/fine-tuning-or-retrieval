num_epochs=100
num_paraphrased=25
yes_paraphrased=_Paraphrased_${num_paraphrased}
echo $num_paraphrased
echo $num_epochs
# nohup python finetuning_knowledge_v6.py --experiment_name ParaphrasedArxivPaper_1B_Full_Finetuning_Test_Run_${num_epochs}_Epochs${yes_paraphrased}_v6_Probes --num_train_epochs $num_epochs --learning_rate 2e-5 --num_paraphrased_texts $num_paraphrased --full_finetuning> output.log 2>&1 &
# sleep 7200
nohup python finetuning_knowledge_v6.py --experiment_name SingleArxivPaper_1B_Full_Finetuning_Test_Run_${num_epochs}_Epochs_v6_Probes --num_train_epochs $num_epochs --learning_rate 2e-5 --num_paraphrased_texts $num_paraphrased --full_finetuning> output_2.log 2>&1 &
