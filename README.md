# NLP From Scratch and Fine-Tuning

This repository contains a collection of NLP projects exploring both **from-scratch implementations** and **fine-tuning of pretrained models**. Each notebook focuses on a different architecture and approach, covering text classification and generative modeling.

## 📂 Notebooks

### 1. CNN + GloVe (AG_NEWS)
- Implements a **1D CNN for text classification**.  
- Uses pretrained **GloVe embeddings** with optional fine-tuning.  
- Dataset: **AG_NEWS** news topic classification.  
- Achieves strong baseline performance for document classification.  

### 2. Character-Level GPT (Hemingway)
- Builds a **mini GPT architecture from scratch in PyTorch**.  
- Includes embeddings, multi-head self-attention, decoder blocks, and generation.  
- Dataset: **Hemingway text corpus**.  
- Demonstrates how generative transformers learn character-level dependencies.  

### 3. GPT-2 Spam Classification
- Fine-tunes **GPT-2** for **binary classification** (SMS Spam vs. Ham).  
- Compares full fine-tuning with frozen embeddings.  
- Dataset: **SMS Spam Collection**.  
- Shows that end-to-end fine-tuning yields >99% accuracy.  

### 4. BERT + LoRA Spam Classification
- Applies **parameter-efficient fine-tuning (LoRA)** on **BERT**.  
- Benchmarks frozen BERT, full fine-tuning, and LoRA adapters.  
- Dataset: **SMS Spam Collection**.  
- LoRA matches full fine-tuning performance (~99% accuracy) while training only ~1.3M parameters.  

## ⚙️ Tech Stack
- **PyTorch**, **torchtext**, **torchmetrics**  
- **Transformers (HuggingFace)**  
- Pretrained embeddings: **GloVe**, **BERT**, **GPT-2**  

## 🚀 Highlights
- Covers both **classic architectures (CNN + embeddings)** and **modern transformers (GPT-2, BERT)**.  
- Includes a **from-scratch GPT** implementation to understand the internals of transformers.  
- Demonstrates **LoRA** for efficient fine-tuning on limited compute.  
