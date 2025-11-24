GPT-Style Transformer From Scratch (100M Parameters)
Overview

This project implements a GPT-style transformer entirely from first principles, including tokenizer, embedding layers, multi-head attention, causal masks, positional encodings, and a full training loop.
It is designed to demonstrate deep understanding of architecture internals rather than relying on high-level frameworks.

Key Features

Custom tokenizer (BPE / WordPiece / Unigram)

Multi-head self-attention implementation

Causal masking (autoregressive)

RoPE-ready positional encoding structure

Transformer blocks with LayerNorm / RMSNorm

Full training loop with AdamW optimizer

Gradient clipping & warmup LR scheduling

Scaling up to 100M+ parameters

Efficient dataloading (WikiText / TinyStories / custom datasets)

Architecture

Tokenizer

Embedding Layer

N × Transformer Blocks

Language Modeling Head

Training

Supports:

FP32 / FP16

Gradient checkpointing

Mini-batch training

Automatic evaluation & loss tracking

Usage
python train.py --dataset WikiText --epochs 10

Future Improvements

FlashAttention

KV-Cache for inference

Grouped-Query Attention (GQA)

Mixture of Experts (MoE) routing

RoPE

LoRA fine-tuning option
