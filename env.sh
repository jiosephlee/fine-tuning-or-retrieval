touch utils/keys.py
mkdir data/olmo/
pip install flash-attn --no-build-isolation
pip install git+https://github.com/huggingface/trl
pip install pydantic datasets==3.6.0 peft bitsandbytes wandb matplotlib seaborn liger-kernel scikit-learn openai
cd ../
git clone https://github.com/jiosephlee/transformers
cd tranformers/
pip install .[torch]
