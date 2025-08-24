num_epochs = 5
nohup python finetuning_knowledge_v6.py --experiment_name SingleArxivPaper_1B_Test_Run_(($num_epochs))_Epochs --num_train_epochs $num_epochs --learning_rate 2e-5 > output.log 2>&1 &