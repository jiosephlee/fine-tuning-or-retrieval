num_epochs=10
<<<<<<< HEAD
nohup python finetuning_knowledge_v6.py --experiment_name ParaphrasedArxivPaper_1B_Full_Finetuning_Test_Run_${num_epochs}_Epochs --num_train_epochs $num_epochs --learning_rate 2e-5 --full_finetuning > output.log 2>&1 &
=======
nohup python finetuning_knowledge_v6.py --experiment_name SingleArxivPaper_1B_Full_Finetuning_Test_Run_${num_epochs}_Epochs --num_train_epochs $num_epochs --learning_rate 2e-5 --full_finetuning > output.log 2>&1 &
>>>>>>> 6001a0863dfc801b84c7433a1835886ad4297cba
