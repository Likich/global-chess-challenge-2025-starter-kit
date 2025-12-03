from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    set_seed,
)
import os
import subprocess
import boto3
from botocore.exceptions import ClientError
from huggingface_hub import login
import torch

from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from torch_xla.core.xla_model import is_master_ordinal
from optimum.neuron.models.training import NeuronModelForCausalLM



def training_function(script_args, training_args):
    dataset = load_dataset("aicrowd/ChessExplained", split="train")
    dataset = dataset.shuffle(seed=23)
    
    # Use appropriate dataset size - adjust based on total available
    # Using smaller subset for workshop timing (~25 minutes training)
    train_dataset = dataset.select(range(min(50000, len(dataset))))
    eval_dataset_end = min(50500, len(dataset))
    eval_dataset_start = min(50000, len(dataset))
    if eval_dataset_end > eval_dataset_start:
        eval_dataset = dataset.select(range(eval_dataset_start, eval_dataset_end))
    else:
        # If dataset is too small, use a small portion of training data for eval
        eval_dataset = dataset.select(range(min(500, len(dataset))))

    def parse_chess_conversation(sample):
        """
        Parse the chess dataset's pre-formatted conversations.
        The 'text' column contains conversations with <|im_start|> and <|im_end|> tags.
        Assistant responses include both <think> reasoning and <uci_move> output.
        """
        text = sample["text"]
        messages = []
        
        # Split by <|im_start|> to get each message block
        parts = text.split("<|im_start|>")
        
        for part in parts[1:]:  # Skip first empty part
            if "<|im_end|>" in part:
                content = part.split("<|im_end|>")[0]
                if content.startswith("user\n"):
                    messages.append({
                        "role": "user",
                        "content": content.replace("user\n", "", 1).strip()
                    })
                elif content.startswith("assistant\n"):
                    messages.append({
                        "role": "assistant", 
                        "content": content.replace("assistant\n", "", 1).strip()
                    })
                elif content.startswith("system\n"):
                    messages.append({
                        "role": "system",
                        "content": content.replace("system\n", "", 1).strip()
                    })
        
        return {"messages": messages}

    train_dataset = train_dataset.map(
        parse_chess_conversation, remove_columns=train_dataset.features, batched=False
    )
    eval_dataset = eval_dataset.map(
        parse_chess_conversation, remove_columns=eval_dataset.features, batched=False
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_id)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.eos_token_id = 128001

    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        script_args.model_id,
        trn_config,
        torch_dtype=dtype,
        # Use FlashAttention2 for better performance and to be able to use larger sequence lengths.
        use_flash_attention_2=False, #Because we are training a sequence lower than 2K for the workshop
    )

    config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "gate_proj",
            "v_proj",
            "o_proj",
            "k_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()

    sft_config = NeuronSFTConfig(
        max_seq_length=1024,
        packing=True,
        **args,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()
    del trainer


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub."
        },
    )
    tokenizer_id: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "The tokenizer used to tokenize text for fine-tuning."},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA r value to be used during fine-tuning."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha value to be used during fine-tuning."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout value to be used during fine-tuning."},
    )
    secret_name: str = field(
        default="huggingface/token",
        metadata={"help": "AWS Secrets Manager secret name containing Hugging Face token."},
    )
    secret_region: str = field(
        default="us-west-2",
        metadata={"help": "AWS region where the secret is stored."},
    )


def get_secret(secret_name, region_name):
    """
    Retrieve a secret from AWS Secrets Manager by searching for secrets with the given name prefix.  
    This is specific to the workshop environment.
    """
    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)
        
        # List secrets and find one that starts with the secret_name
        paginator = client.get_paginator('list_secrets')
        for page in paginator.paginate():
            for secret in page['SecretList']:
                if secret['Name'].startswith(secret_name):
                    response = client.get_secret_value(SecretId=secret['ARN'])
                    if 'SecretString' in response:
                        return response['SecretString']
        return None
    except ClientError:
        print("Could not retrieve secret from AWS Secrets Manager")
        return None

if __name__ == "__main__":
    parser = HfArgumentParser([ScriptArguments, NeuronTrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # Check for Hugging Face token in environment variable
    hf_token = os.environ.get("HF_TOKEN")
    
    # If no token in environment, try to get it from AWS Secrets Manager
    if not hf_token:
        print("No Hugging Face token found in environment, checking AWS Secrets Manager...")
        hf_token = get_secret(script_args.secret_name, script_args.secret_region)
    
    # Login to Hugging Face if a valid token is found
    if hf_token:
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
    else:
        print("No valid Hugging Face token found, continuing without authentication")
    
    set_seed(training_args.seed)
    training_function(script_args, training_args)

    # Consolidate LoRA adapter shards, merge LoRA adapters into base model, save merged model
    if is_master_ordinal():
        input_ckpt_dir = os.path.join(
            training_args.output_dir, f"checkpoint-{training_args.max_steps}"
        )
        output_ckpt_dir = os.path.join(training_args.output_dir, "merged_model")
        # the spawned process expects to see 2 NeuronCores for consolidating checkpoints with a tp=2
        # Either the second core isn't really used or it is freed up by the other thread finishing.  
        # Adjusting Neuron env. var to advertise 2 NeuronCores to the process.
        env = os.environ.copy()
        env["NEURON_RT_VISIBLE_CORES"] = "0-1"
        subprocess.run(
            [
                "python3",
                "consolidate_adapter_shards_and_merge_model.py",
                "-i",
                input_ckpt_dir,
                "-o",
                output_ckpt_dir,
            ],
            env=env
        )

