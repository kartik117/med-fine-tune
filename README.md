# MedFineTune

QLoRA fine-tuning of Mistral-7B-Instruct on clinical QA datasets (PubMedQA + MedQA), served through a vLLM OpenAI-compatible endpoint.

## Approach

```
PubMedQA + MedQA (Hugging Face) -> QLoRA fine-tune (4-bit NF4, r=64, lora_alpha=128)
                                       on Mistral-7B-Instruct-v0.2 (Colab A100 / Kaggle T4)
                                          -> push adapter to Hugging Face Hub
                                              -> serve via vLLM
                                                  -> evaluate with lm-evaluation-harness, track in W&B
```

QLoRA config: `r=64, lora_alpha=128, lora_dropout=0.05, target_modules=["q_proj","v_proj"]`

## Stack

PyTorch · PEFT + bitsandbytes (QLoRA) · Hugging Face Hub · vLLM · Weights & Biases · lm-evaluation-harness
