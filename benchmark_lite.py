"""
Benchmark simplificado de modelos Transformer para hardware limitado.
Inclui: DistilBERT, DistilGPT2, BERT-Tiny, TinyLLaMA, Gemma 2 Mini (tenta quantizar em 4-bit).
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import evaluate
import mlflow
import mlflow.pytorch
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from collections import Counter

# Try to import BitsAndBytesConfig (for 4-bit quantization). Fallback if unavailable.
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False

# Detecta dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device em uso: {device}")

# Configuração do MLflow
mlflow.set_experiment("Benchmark_Transformers_SB100_Lite")


def get_model_size(model):
    """Retorna o tamanho do modelo em MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def safe_load_metric(name):
    """Carrega métrica do evaluate, com fallback."""
    try:
        return evaluate.load(name)
    except Exception as e:
        print(f"Métrica '{name}' indisponível: {e}")
        return None


def safe_compute_bleu(predictions, references):
    """Calcula BLEU via evaluate ou sacrebleu."""
    try:
        bleu = safe_load_metric('bleu')
        if bleu:
            return bleu.compute(predictions=predictions, references=[[r] for r in references])
    except Exception:
        pass
    try:
        import sacrebleu
        score = sacrebleu.corpus_bleu(predictions, [references])
        return {"bleu": float(score.score / 100.0)}
    except Exception:
        return {"bleu": 0.0}


def safe_compute_rouge(predictions, references):
    """Calcula ROUGE-L via evaluate ou rouge_score fallback."""
    try:
        rouge = safe_load_metric('rouge')
        if rouge:
            return rouge.compute(predictions=predictions, references=references)
    except Exception:
        pass
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        vals = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(predictions, references)]
        return {"rougeL": float(np.mean(vals) if vals else 0.0)}
    except Exception:
        return {"rougeL": 0.0}


def compute_bleu(predictions, references):
    try:
        import sacrebleu
        return float(sacrebleu.corpus_bleu(predictions, [references]).score / 100.0)
    except Exception:
        try:
            bleu = safe_load_metric('bleu')
            if bleu:
                res = bleu.compute(predictions=predictions, references=[[r] for r in references])
                return float(res.get('bleu', 0.0))
        except Exception:
            pass
    return 0.0


def compute_rouge_l(predictions, references):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(predictions, references)]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        try:
            rouge = safe_load_metric('rouge')
            if rouge:
                res = rouge.compute(predictions=predictions, references=references)
                return float(res.get('rougeL', 0.0))
        except Exception:
            pass
    return 0.0


def load_causal_model(model_name, quantize_4bit=True):
    """
    Carrega model CausalLM com tentativa de quantização 4-bit.
    Retorna tokenizer, model, load_time, used_quantization(boolean).
    """
    start_load = time.time()
    used_q = False

    if quantize_4bit and BNB_AVAILABLE and device == "cuda":
        try:
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=q_config,
                device_map="auto",
                trust_remote_code=True
            )
            used_q = True
        except Exception as e:
            print(f"Falha ao carregar quantizado (4-bit) {model_name}: {e}. Tentando sem quantização.")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            used_q = False
    else:
        # fallback normal load
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        used_q = False

    load_time = time.time() - start_load
    return tokenizer, model, load_time, used_q


def load_seqcls_model(model_name, quantize_4bit=False):
    """
    Carrega modelo de Sequence Classification.
    Quantização 4-bit geralmente não é usada para seqcls aqui (fallback).
    """
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).to(device)
    load_time = time.time() - start_load
    return tokenizer, model, load_time


def benchmark_distilbert(do_train=False):
    """DistilBERT - classificação (AG News small)."""
    print("\nIniciando teste com DistilBERT...")

    with mlflow.start_run(run_name="DistilBERT_lite"):
        model_name = "distilbert-base-uncased"

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "classification")
        mlflow.log_param("dataset", "ag_news[:100]")
        mlflow.log_param("device", device)

        dataset = load_dataset("ag_news", split="train[:100]")
        test_dataset = load_dataset("ag_news", split="test[:25]")

        print("Carregando modelo...")
        tokenizer, model, load_time = load_seqcls_model(model_name)

        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)

        mlflow.log_metric("num_parameters", num_params)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_metric("load_time_seconds", load_time)
        mlflow.log_metric("training_time_seconds", 0.0)

        if do_train:
            print("Executando fine-tune rápido (1 epoch)...")
            train_texts = [x['text'] for x in dataset][:50]
            train_labels = [x['label'] for x in dataset][:50]

            enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
            input_ids, mask = enc['input_ids'], enc['attention_mask']
            labels_tensor = torch.tensor(train_labels)

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            model.train()
            start_train = time.time()
            for i in range(0, len(train_texts), 8):
                batch_ids = input_ids[i:i+8].to(device)
                batch_mask = mask[i:i+8].to(device)
                batch_labels = labels_tensor[i:i+8].to(device)

                outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_time = time.time() - start_train
            mlflow.log_metric("training_time_seconds", train_time)
            print(f"Fine-tune: {train_time:.2f}s")

        texts = [x["text"] for x in test_dataset if x.get("text")]
        labels = [x["label"] for x in test_dataset if x.get("text")]

        preds = []
        start_eval = time.time()

        try:
            pipe_cls = pipeline("text-classification", model=model_name, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
            use_pipe = True
        except Exception:
            use_pipe = False
            pipe_cls = None

        for t in texts:
            if use_pipe and pipe_cls is not None:
                try:
                    out = pipe_cls(t[:512])
                    label = out[0]['label']
                    pred = int(label.split('_')[1]) if label.startswith('LABEL_') else int(label)
                except Exception:
                    inputs_eval = tokenizer(t[:512], return_tensors='pt', truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        logits = model(**inputs_eval).logits
                    pred = int(torch.argmax(logits, dim=-1).cpu().item())
            else:
                inputs_eval = tokenizer(t[:512], return_tensors='pt', truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    logits = model(**inputs_eval).logits
                pred = int(torch.argmax(logits, dim=-1).cpu().item())

            preds.append(pred)

        eval_time = time.time() - start_eval

        accuracy = accuracy_score(labels, preds) if labels and preds else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0) if labels and preds else (0.0, 0.0, 0.0, None)

        mlflow.log_metric("inference_time_seconds", eval_time)
        mlflow.log_metric("inference_time_per_sample_seconds", eval_time / len(texts) if texts else 0.0)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        print(f"Acurácia: {accuracy:.4f} | F1: {f1:.4f} | Inferência: {eval_time:.2f}s")

        pred_counter = Counter(preds)
        print(f"Distribuição: {pred_counter}")

        try:
            cm = confusion_matrix(labels, preds)
            if plt and sns:
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Matriz de confusão - DistilBERT")
                plt.xlabel("Predição")
                plt.ylabel("Verdadeiro")
                cm_path = "confusion_matrix_distilbert.png"
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)
        except Exception as e:
            print(f"Erro matriz confusão: {e}")

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "model": "DistilBERT",
            "task": "classification",
            "accuracy": accuracy,
            "f1_score": f1,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "inference_time_seconds": eval_time
        }


def benchmark_distilgpt2():
    """DistilGPT2 - geração (Wikitext small)."""
    print("\nIniciando teste com DistilGPT2...")

    with mlflow.start_run(run_name="DistilGPT2_lite"):
        model_name = "distilgpt2"
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "text-generation")
        mlflow.log_param("dataset", "wikitext-2-raw-v1[:10]")
        mlflow.log_param("device", device)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10]")

        tokenizer, model, load_time, used_q = load_causal_model(model_name, quantize_4bit=False)  # distilgpt2 small - no quant forced
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)

        mlflow.log_metric("num_parameters", num_params)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_metric("load_time_seconds", load_time)

        prompts = ["The future of robotics is", "Data science will improve"]
        preds, refs = [], []
        start_inf = time.time()

        for p in prompts:
            enc = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=40, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            preds.append(text)
            refs.append(p)

        inf_time = time.time() - start_inf

        bleu = compute_bleu(preds, refs)
        rougeL = compute_rouge_l(preds, refs)

        mlflow.log_metric("inference_time_seconds", inf_time)
        mlflow.log_metric("bleu", bleu)
        mlflow.log_metric("rougeL", rougeL)

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "model": "DistilGPT2",
            "task": "text-generation",
            "bleu": bleu,
            "rougeL": rougeL,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "inference_time_seconds": inf_time
        }


def benchmark_bert_tiny():
    """BERT-tiny - classificação (AG News small)."""
    print("\nIniciando teste com BERT Tiny...")

    with mlflow.start_run(run_name="BERT_Tiny_lite"):
        model_name = "prajjwal1/bert-tiny"
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "classification")
        mlflow.log_param("dataset", "ag_news[:100]")
        mlflow.log_param("device", device)

        dataset = load_dataset("ag_news", split="train[:100]")
        test_dataset = load_dataset("ag_news", split="test[:25]")

        tokenizer, model, load_time = load_seqcls_model(model_name)

        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)

        mlflow.log_metric("num_parameters", num_params)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_metric("load_time_seconds", load_time)
        mlflow.log_metric("training_time_seconds", 0.0)

        texts = [x["text"] for x in test_dataset if x.get("text")]
        labels = [x["label"] for x in test_dataset if x.get("text")]

        preds = []
        start_eval = time.time()

        for t in texts:
            enc = tokenizer(t, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            preds.append(int(torch.argmax(logits, dim=-1).cpu().item()))

        eval_time = time.time() - start_eval

        accuracy = accuracy_score(labels, preds) if labels and preds else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0) if labels and preds else (0.0, 0.0, 0.0, None)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("inference_time_seconds", eval_time)

        print(f"Acurácia: {accuracy:.4f} | F1: {f1:.4f} | Inferência: {eval_time:.2f}s")

        try:
            cm = confusion_matrix(labels, preds)
            if plt and sns:
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Matriz de confusão - BERT Tiny")
                cm_path = "confusion_matrix_bert_tiny.png"
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)
        except Exception as e:
            print(f"Erro matriz confusão: {e}")

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "model": "BERT_Tiny",
            "task": "classification",
            "accuracy": accuracy,
            "f1_score": f1,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "inference_time_seconds": eval_time
        }


def benchmark_tiny_llama(quantize_4bit=True):
    """TinyLLaMA - geração - tenta 4-bit se disponível."""
    print("\nIniciando teste com TinyLLaMA...")

    with mlflow.start_run(run_name="TinyLLaMA_lite"):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "text-generation")
        mlflow.log_param("dataset", "wikitext-2-raw-v1[:10]")
        mlflow.log_param("device", device)
        mlflow.log_param("quantize_4bit_enabled", bool(quantize_4bit and BNB_AVAILABLE and device == 'cuda'))

        tokenizer, model, load_time, used_q = load_causal_model(model_name, quantize_4bit=quantize_4bit and BNB_AVAILABLE and device == 'cuda')

        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)
        mlflow.log_metric("num_parameters", num_params)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_metric("load_time_seconds", load_time)

        prompts = [
            "O futuro da agricultura é",
            "As mudanças climáticas afetam a agricultura ao"
        ]
        preds, refs = [], []
        start_inf = time.time()

        for p in prompts:
            enc = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=60, do_sample=True, top_p=0.9)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            preds.append(text)
            refs.append(p)

        inf_time = time.time() - start_inf

        bleu = compute_bleu(preds, refs)
        rougeL = compute_rouge_l(preds, refs)

        mlflow.log_metric("inference_time_seconds", inf_time)
        mlflow.log_metric("bleu", bleu)
        mlflow.log_metric("rougeL", rougeL)

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "model": "TinyLLaMA",
            "task": "text-generation",
            "bleu": bleu,
            "rougeL": rougeL,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "inference_time_seconds": inf_time,
            "used_quantization_4bit": used_q
        }


def benchmark_gemma2_mini(quantize_4bit=True):
    """Gemma 2 Mini - geração - tenta 4-bit se disponível."""
    print("\nIniciando teste com Gemma 2 Mini...")

    with mlflow.start_run(run_name="Gemma2_Mini_lite"):
        model_name = "google/gemma-2-2b-mini"
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task", "text-generation")
        mlflow.log_param("dataset", "wikitext-2-raw-v1[:10]")
        mlflow.log_param("device", device)
        mlflow.log_param("quantize_4bit_enabled", bool(quantize_4bit and BNB_AVAILABLE and device == 'cuda'))

        tokenizer, model, load_time, used_q = load_causal_model(model_name, quantize_4bit=quantize_4bit and BNB_AVAILABLE and device == 'cuda')

        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)
        mlflow.log_metric("num_parameters", num_params)
        mlflow.log_metric("model_size_mb", model_size_mb)
        mlflow.log_metric("load_time_seconds", load_time)

        prompts = [
            "O futuro da agricultura é",
            "As mudanças climáticas afetam o cultivo ao"
        ]
        preds, refs = [], []
        start_inf = time.time()

        for p in prompts:
            enc = tokenizer(p, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=60, do_sample=True, top_p=0.9)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            preds.append(text)
            refs.append(p)

        inf_time = time.time() - start_inf

        bleu = compute_bleu(preds, refs)
        rougeL = compute_rouge_l(preds, refs)

        mlflow.log_metric("inference_time_seconds", inf_time)
        mlflow.log_metric("bleu", bleu)
        mlflow.log_metric("rougeL", rougeL)

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return {
            "model": "Gemma2_Mini",
            "task": "text-generation",
            "bleu": bleu,
            "rougeL": rougeL,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "inference_time_seconds": inf_time,
            "used_quantization_4bit": used_q
        }


def main():
    """Execução principal do benchmark."""
    header = """
    BENCHMARK SIMPLIFICADO - Modelos Transformer (Versão Lite)
    Projeto CCD SB100 – Squad 4
    Otimizado para notebooks com recursos limitados
    """
    print(header)

    parser = argparse.ArgumentParser(description='Benchmarks lite for transformer models')
    parser.add_argument('--model', choices=['all','distilbert','distilgpt2','bert_tiny','tinyllama','gemma2mini'], default='all', help='Which model to run')
    parser.add_argument('--fine_tune', action='store_true', help='Run a quick fine-tuning step (1 epoch) before evaluation for classification models')
    parser.add_argument('--no_quant', action='store_true', help='Disable 4-bit quantization attempts')
    args = parser.parse_args()

    selected = args.model
    do_train = args.fine_tune
    allow_quant = not args.no_quant

    results = []

    try:
        if selected in ['all','distilbert']:
            results.append(benchmark_distilbert(do_train=do_train))
    except Exception as e:
        print(f"Erro DistilBERT: {e}")

    try:
        if selected in ['all','distilgpt2']:
            results.append(benchmark_distilgpt2())
    except Exception as e:
        print(f"Erro DistilGPT2: {e}")

    try:
        if selected in ['all','bert_tiny']:
            results.append(benchmark_bert_tiny())
    except Exception as e:
        print(f"Erro BERT Tiny: {e}")

    try:
        if selected in ['all','tinyllama']:
            results.append(benchmark_tiny_llama(quantize_4bit=allow_quant))
    except Exception as e:
        print(f"Erro TinyLLaMA: {e}")

    try:
        if selected in ['all','gemma2mini']:
            results.append(benchmark_gemma2_mini(quantize_4bit=allow_quant))
    except Exception as e:
        print(f"Erro Gemma2 Mini: {e}")

    # Salva resultados
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_lite_{timestamp}.csv"
        df.to_csv(filename, index=False)

        print("\nRESUMO DOS RESULTADOS")
        print(df.to_string(index=False))
        print(f"\nSalvo em: {filename}")
        print("\nPara visualizar no MLflow, execute: mlflow ui")

    print("\nBenchmark concluído.")


if __name__ == "__main__":
    main()
