from llava.train.train import train
import os
os.environ["WANDB_API_KEY"] = '7352f9a349b74e01062672d0bc0bd3a8094677e2' 
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
