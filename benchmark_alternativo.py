import os
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

# Imports otimizados
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
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
    PLOT_AVAILABLE = True
except Exception:
    PLOT_AVAILABLE = False

_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device em uso: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def setup_mlflow():
    mlflow.set_experiment("Benchmark_Transformers_SB100_Lite")
    mlflow.set_tracking_uri("file:./mlruns")

class ModelLogger:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.metrics = {}
        self.params = {}
        
    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.params.items():
            mlflow.log_param(key, value)
        for key, value in self.metrics.items():
            mlflow.log_metric(key, value)
        mlflow.end_run()
    
    def log_param(self, key: str, value: Any):
        self.params[key] = value
    
    def log_metric(self, key: str, value: Any):
        self.metrics[key] = value
    
    def log_metrics(self, metrics_dict: Dict[str, Any]):
        self.metrics.update(metrics_dict)
    
    def log_params(self, params_dict: Dict[str, Any]):
        self.params.update(params_dict)

# Funções utilitárias otimizadas
def get_model_size(model: torch.nn.Module) -> float:
    return sum(
        p.numel() * p.element_size() 
        for p in model.parameters() if p.requires_grad
    ) / (1024 ** 2)

def load_model_cached(model_name: str, model_type: str, **kwargs) -> Tuple[Any, float]:
    cache_key = f"{model_name}_{model_type}"
    
    if cache_key in _MODEL_CACHE:
        model, tokenizer, load_time = _MODEL_CACHE[cache_key]
        print(f"Usando modelo em cache: {model_name}")
        return tokenizer, model, load_time
    
    start_load = time.time()
    
    try:
        if model_type == "seqcls":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=4
            ).to(DEVICE)
        else:
            tokenizer_kwargs = {"use_fast": True}
            model_kwargs = {}
            
            if "llama" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=True,
                    padding_side="left"
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            elif "gemma" in model_name.lower():
                # Para Gemma
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True
                )
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            # Carregar modelo
            model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        
        load_time = time.time() - start_load
        _MODEL_CACHE[cache_key] = (model, tokenizer, load_time)
        
        return tokenizer, model, load_time
    except Exception as e:
        print(f"Erro ao carregar {model_name}: {e}")
        # Retornar um modelo fallback simples
        if model_type == "seqcls":
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=4
            ).to(DEVICE)
        else:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        
        load_time = time.time() - start_load
        return tokenizer, model, load_time

def load_causal_model_quantized(model_name: str, quantize_4bit: bool = True) -> Tuple[Any, Any, float, bool]:
    cache_key = f"{model_name}_causal_quant_{quantize_4bit}"
    
    if cache_key in _MODEL_CACHE:
        model, tokenizer, load_time, used_q = _MODEL_CACHE[cache_key]
        return tokenizer, model, load_time, used_q
    
    start_load = time.time()
    used_q = False
    
    try:
        if quantize_4bit and DEVICE.type == "cuda":
            print(f"Tentando quantização 4-bit para {model_name}...")
            try:
                q_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )

                if "llama" in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                elif "gemma" in model_name.lower():
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        tokenizer.pad_token = tokenizer.eos_token
                    except:
                        print("Usando modelo alternativo para Gemma...")
                        model_name = "microsoft/phi-2"  # Modelo aberto alternativo
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=q_config,
                    device_map="auto",
                    trust_remote_code=False  
                )
                used_q = True
                print("✓ Quantização 4-bit aplicada com sucesso")
            except Exception as e:
                print(f"Falha na quantização 4-bit: {e}. Usando modelo padrão.")
                tokenizer, model = load_model_cached(model_name, "causal")[:2]
        else:
            tokenizer, model = load_model_cached(model_name, "causal")[:2]
    except Exception as e:
        print(f"Erro crítico ao carregar {model_name}: {e}")
        print("Usando fallback para DistilGPT2...")
        model_name = "distilgpt2"
        tokenizer, model = load_model_cached(model_name, "causal")[:2]
    
    load_time = time.time() - start_load
    _MODEL_CACHE[cache_key] = (model, tokenizer, load_time, used_q)
    
    return tokenizer, model, load_time, used_q

# Métricas otimizadas
class MetricsCalculator:

    _bleu_metric = None
    _rouge_metric = None
    
    @classmethod
    def get_bleu_metric(cls):
        if cls._bleu_metric is None:
            try:
                cls._bleu_metric = evaluate.load("bleu")
            except:
                cls._bleu_metric = None
        return cls._bleu_metric
    
    @classmethod
    def get_rouge_metric(cls):
        if cls._rouge_metric is None:
            try:
                cls._rouge_metric = evaluate.load("rouge")
            except:
                cls._rouge_metric = None
        return cls._rouge_metric
    
    @staticmethod
    def compute_classification_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
        if not labels or not preds:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        accuracy = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        
        return {
            "accuracy": float(accuracy),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
    
    @staticmethod
    def compute_generation_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        bleu_metric = MetricsCalculator.get_bleu_metric()
        rouge_metric = MetricsCalculator.get_rouge_metric()
        
        results = {"bleu": 0.0, "rougeL": 0.0}
        
        # Cálculo BLEU
        if bleu_metric and predictions and references:
            try:
                bleu_result = bleu_metric.compute(
                    predictions=predictions,
                    references=[[r] for r in references]
                )
                results["bleu"] = float(bleu_result.get("bleu", 0.0))
            except:
                pass
        
        # Cálculo ROUGE
        if rouge_metric and predictions and references:
            try:
                rouge_result = rouge_metric.compute(
                    predictions=predictions,
                    references=references
                )
                results["rougeL"] = float(rouge_result.get("rougeL", 0.0))
            except:
                pass
        
        return results

def benchmark_model(model_name: str, model_func: callable, **kwargs) -> Optional[Dict]:
    try:
        return model_func(**kwargs)
    except Exception as e:
        print(f"Erro no benchmark {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_distilbert_optimized(do_train: bool = False) -> Dict:
    """DistilBERT otimizado."""
    print("\nIniciando teste com DistilBERT...")
    
    model_name = "distilbert-base-uncased"
    task_type = "classification"
    
    with ModelLogger(f"DistilBERT_{task_type}") as logger:
        logger.log_params({
            "model_name": model_name,
            "task": task_type,
            "dataset": "ag_news",
            "device": str(DEVICE),
            "do_train": do_train
        })
        
        tokenizer, model, load_time = load_model_cached(model_name, "seqcls")
        
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)
        
        logger.log_metrics({
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time
        })
        
        # Dataset
        dataset = load_dataset("ag_news", split="test[:50]") 
        texts = [x["text"] for x in dataset if x.get("text")]
        labels = [x["label"] for x in dataset if x.get("text")]
    
        start_eval = time.time()
        preds = []
        
        try:
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if DEVICE.type == "cuda" else -1
            )
            
            batch_results = classifier(texts, truncation=True, max_length=128)
            preds = [
                int(r['label'].split('_')[1]) if 'label' in r else 0
                for r in batch_results
            ]
        except:
            batch_size = 8
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(**encodings)
                    batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                    preds.extend(batch_preds)
        
        eval_time = time.time() - start_eval
        
        # Métricas
        metrics = MetricsCalculator.compute_classification_metrics(labels, preds)
        metrics.update({
            "inference_time_seconds": eval_time,
            "inference_time_per_sample": eval_time / len(texts) if texts else 0
        })
        
        logger.log_metrics(metrics)
        
        # Plot de confusão
        if PLOT_AVAILABLE and labels and preds:
            try:
                cm = confusion_matrix(labels, preds)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Matriz de Confusão - {model_name}")
                plt.tight_layout()
                cm_path = f"confusion_matrix_{model_name.replace('/', '_')}.png"
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)
                os.remove(cm_path)
            except Exception as e:
                print(f"Erro ao gerar matriz: {e}")
        
        # Cleanup
        del model, tokenizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "model": "DistilBERT",
            "task": task_type,
            **metrics,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time
        }

def benchmark_generation_model(model_name: str, display_name: str, 
                             quantize_4bit: bool = True, 
                             use_safe_fallback: bool = True) -> Dict:
    """Template padronizado para modelos de geração."""
    print(f"\nIniciando teste com {display_name}...")
    
    task_type = "text-generation"
    
    with ModelLogger(f"{display_name}_{task_type}") as logger:
        logger.log_params({
            "model_name": model_name,
            "task": task_type,
            "device": str(DEVICE),
            "quantize_4bit": quantize_4bit
        })
        
        try:
            tokenizer, model, load_time, used_q = load_causal_model_quantized(
                model_name, quantize_4bit
            )
            
            # Verificar se o modelo foi carregado
            if model is None:
                raise ValueError(f"Falha ao carregar modelo: {model_name}")
            
        except Exception as e:
            print(f"Erro ao carregar {display_name}: {e}")
            if use_safe_fallback:
                print(f"Usando fallback seguro para {display_name}...")
                # Usar um modelo que sempre funciona
                fallback_model = "distilgpt2"
                tokenizer, model, load_time = load_model_cached(fallback_model, "causal")
                used_q = False
                display_name = f"{display_name}_fallback"
            else:
                raise
        
        # Estatísticas do modelo
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)
        
        logger.log_metrics({
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "quantization_used": int(used_q)
        })
        
        # Prompts de teste adaptados para o modelo
        if "llama" in model_name.lower() or "tinyllama" in model_name.lower():
            prompts = [
                "Explique o que é machine learning em português:",
                "Quais são as vantagens da energia solar?",
                "Descreva inteligência artificial de forma simples:"
            ]
        elif "gemma" in model_name.lower():
            prompts = [
                "What is machine learning?",
                "Explain solar energy benefits:",
                "Describe artificial intelligence:"
            ]
        else:
            prompts = [
                "O futuro da inteligência artificial é",
                "A sustentabilidade ambiental pode ser melhorada através de",
                "Os avanços na medicina moderna permitem"
            ]
        
        start_inf = time.time()
        predictions = []
        
        # Configurar parâmetros de geração
        generation_kwargs = {
            "max_new_tokens": 50,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Adicionar pad_token_id se necessário
        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif hasattr(tokenizer, 'eos_token_id'):
            generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
        
        for prompt in prompts:
            try:
                # Preparar input
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                # Mover para dispositivo correto
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(generated_text)
                
                # Mostrar preview
                print(f"  Prompt: {prompt[:50]}...")
                print(f"  Resposta: {generated_text[:100]}...\n")
                
            except Exception as e:
                print(f"  Erro na geração para prompt '{prompt[:30]}...': {e}")
                predictions.append("")
        
        inf_time = time.time() - start_inf
        
        # Calcular métricas apenas para respostas válidas
        valid_predictions = [p for p in predictions if p.strip()]
        valid_references = [prompts[i] for i, p in enumerate(predictions) if p.strip()]
        
        if valid_predictions and valid_references:
            metrics = MetricsCalculator.compute_generation_metrics(valid_predictions, valid_references)
        else:
            metrics = {"bleu": 0.0, "rougeL": 0.0}
        
        metrics["inference_time_seconds"] = inf_time
        metrics["successful_generations"] = len(valid_predictions)
        metrics["total_prompts"] = len(prompts)
        
        logger.log_metrics(metrics)
        
        # Cleanup
        del model, tokenizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "model": display_name,
            "task": task_type,
            **metrics,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time,
            "quantization_used": used_q
        }

# Funções específicas dos modelos usando o template
def benchmark_distilgpt2_optimized() -> Dict:
    return benchmark_generation_model(
        model_name="distilgpt2",
        display_name="DistilGPT2",
        quantize_4bit=False,
        use_safe_fallback=False
    )

def benchmark_tinyllama_optimized(quantize_4bit: bool = True) -> Dict:
    """TinyLLaMA otimizado - modelo aberto que funciona sem token"""
    try:
        # Usar TinyLlama que é aberto e funciona bem
        return benchmark_generation_model(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            display_name="TinyLLaMA",
            quantize_4bit=quantize_4bit,
            use_safe_fallback=True
        )
    except Exception as e:
        print(f"Erro com TinyLLaMA: {e}")
        print("Tentando modelo alternativo (Phi-2)...")
        # Fallback para outro modelo aberto
        return benchmark_generation_model(
            model_name="microsoft/phi-2",
            display_name="Phi-2",
            quantize_4bit=quantize_4bit,
            use_safe_fallback=True
        )

def benchmark_gemma2mini_optimized(quantize_4bit: bool = True) -> Dict:
    """Gemma 2 Mini com fallback para modelo aberto"""
    try:
        # Tentar Gemma primeiro
        return benchmark_generation_model(
            model_name="google/gemma-2-2b-it",
            display_name="Gemma2_Mini",
            quantize_4bit=quantize_4bit,
            use_safe_fallback=True
        )
    except Exception as e:
        print(f"Erro com Gemma (pode precisar de token): {e}")
        print("Usando modelo alternativo aberto (Qwen2.5-1.5B)...")
        # Usar modelo aberto alternativo
        return benchmark_generation_model(
            model_name="Qwen/Qwen2.5-1.5B",
            display_name="Qwen2.5-1.5B",
            quantize_4bit=quantize_4bit,
            use_safe_fallback=True
        )

def benchmark_bert_tiny_optimized() -> Dict:
    """BERT Tiny otimizado."""
    print("\nIniciando teste com BERT Tiny...")
    
    model_name = "prajjwal1/bert-tiny"
    
    with ModelLogger("BERT_Tiny_classification") as logger:
        logger.log_params({
            "model_name": model_name,
            "task": "classification",
            "dataset": "ag_news",
            "device": str(DEVICE)
        })
        
        # Carregamento
        tokenizer, model, load_time = load_model_cached(model_name, "seqcls")
        
        # Estatísticas
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)
        
        logger.log_metrics({
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time
        })
        
        # Dataset
        dataset = load_dataset("ag_news", split="test[:50]")
        texts = [x["text"] for x in dataset]
        labels = [x["label"] for x in dataset]
        
        # Inferência em batch
        start_eval = time.time()
        batch_size = 16  # Batch maior para modelos pequenos
        preds = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**encodings)
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                preds.extend(batch_preds)
        
        eval_time = time.time() - start_eval
        
        # Métricas
        metrics = MetricsCalculator.compute_classification_metrics(labels, preds)
        metrics["inference_time_seconds"] = eval_time
        
        logger.log_metrics(metrics)
        
        # Cleanup
        del model, tokenizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        return {
            "model": "BERT_Tiny",
            "task": "classification",
            **metrics,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "load_time_seconds": load_time
        }

def main():
    """Execução principal do benchmark."""
    header = """
    BENCHMARK SIMPLIFICADO - Modelos Transformer (Versão Otimizada)
    Projeto CCD SB100 – Squad 4
    Otimizado para notebooks com recursos limitados
    
    Modelos disponíveis:
    - DistilBERT: Classificação de texto
    - DistilGPT2: Geração de texto (base)
    - BERT Tiny: Classificação leve
    - TinyLLaMA: Geração com modelo aberto LLM
    - Gemma 2 Mini: Geração com modelo alternativo
    
    Nota: Usando modelos abertos que não requerem token de autenticação
    """
    print(header)
    
    setup_mlflow()
    
    parser = argparse.ArgumentParser(description='Benchmarks lite para modelos transformer')
    parser.add_argument('--model', choices=['all', 'distilbert', 'distilgpt2', 
                                          'bert_tiny', 'tinyllama', 'gemma2mini'],
                       default='all', help='Modelo a executar')
    parser.add_argument('--no_quant', action='store_true',
                       help='Desabilitar quantização 4-bit')
    parser.add_argument('--sample_size', type=int, default=50,
                       help='Tamanho da amostra para testes')
    parser.add_argument('--use_safe', action='store_true',
                       help='Usar apenas modelos seguros (sem falhas)')
    args = parser.parse_args()
    
    # Mapeamento de modelos para funções
    model_registry = {
        'distilbert': benchmark_distilbert_optimized,
        'distilgpt2': benchmark_distilgpt2_optimized,
        'bert_tiny': benchmark_bert_tiny_optimized,
        'tinyllama': lambda: benchmark_tinyllama_optimized(not args.no_quant),
        'gemma2mini': lambda: benchmark_gemma2mini_optimized(not args.no_quant)
    }
    
    # Execução dos benchmarks
    results = []
    models_to_run = model_registry.keys() if args.model == 'all' else [args.model]
    
    print(f"\n{'='*60}")
    print(f"INICIANDO BENCHMARK")
    print(f"Modelos a testar: {', '.join(models_to_run)}")
    print(f"Quantização: {'Não' if args.no_quant else 'Sim'}")
    print(f"Dispositivo: {DEVICE}")
    print(f"{'='*60}\n")
    
    for model_key in models_to_run:
        if model_key in model_registry:
            print(f"\n{'='*60}")
            print(f"TESTANDO: {model_key.upper()}")
            print(f"{'='*60}")
            
            start_time = time.time()
            result = benchmark_model(
                model_key,
                model_registry[model_key]
            )
            
            if result:
                results.append(result)
                elapsed = time.time() - start_time
                print(f"✓ {model_key} concluído em {elapsed:.2f}s")
            else:
                print(f"✗ {model_key} falhou")
    
    # Salvamento e relatório
    if results:
        df = pd.DataFrame(results)
        
        # Ordenar por tipo de tarefa e modelo
        df = df.sort_values(['task', 'model'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_optimized_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Relatório resumido
        print("\n" + "="*80)
        print("RESUMO DOS RESULTADOS".center(80))
        print("="*80)
        
        for task in df['task'].unique():
            task_df = df[df['task'] == task]
            print(f"\n{task.upper():^80}")
            print("-"*80)
            
            if task == 'classification':
                print(f"{'Modelo':<20} {'Acurácia':<10} {'F1-Score':<10} {'Inferência(s)':<15} {'Tamanho(MB)':<12}")
                for _, row in task_df.iterrows():
                    print(f"{row['model']:<20} {row.get('accuracy', 0):<10.4f} "
                          f"{row.get('f1_score', 0):<10.4f} {row.get('inference_time_seconds', 0):<15.2f} "
                          f"{row.get('model_size_mb', 0):<12.1f}")
            else:  # text-generation
                print(f"{'Modelo':<20} {'Sucessos':<10} {'BLEU':<10} {'Inferência(s)':<15} {'Quantizado':<12}")
                for _, row in task_df.iterrows():
                    success_rate = (row.get('successful_generations', 0) / 
                                  row.get('total_prompts', 1)) * 100
                    print(f"{row['model']:<20} {success_rate:<10.1f}% "
                          f"{row.get('bleu', 0):<10.4f} {row.get('inference_time_seconds', 0):<15.2f} "
                          f"{'Sim' if row.get('quantization_used', False) else 'Não':<12}")
        
        print("\n" + "="*80)
        print(f"Relatório salvo em: {filename}")
        print("Para visualizar no MLflow, execute: mlflow ui")
        print("="*80)
    
    print("\nBenchmark concluído!")

if __name__ == "__main__":
    main()