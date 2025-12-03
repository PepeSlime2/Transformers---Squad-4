## Guia rápido

Siga estes passos para rodar o benchmark leve no seu notebook.

1) Criar ambiente virtual e ativar
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2) Instalar dependências
```powershell
pip install -r requirements.txt
```

3) Rodar benchmark leve
```powershell
python benchmark_lite.py
```

4) Visualizar resultados (opcional)
```powershell
mlflow ui --host 127.0.0.1 --port 5001
```
Abra: http://127.0.0.1:5001

Restaurar backup (se necessário)
```powershell
Expand-Archive -Path restore_files.zip -DestinationPath . -Force
```

Se precisar de instruções mais detalhadas, consulte o `README.md`.

```markdown
# Guia Rápido de Instalação e Uso

Este guia leva você do zero até rodar um benchmark rápido.

Pré-requisitos
- Python 3.10+ instalado
- Internet para baixar modelos e datasets

1) Criar e ativar ambiente virtual
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2) Instalar dependências
```powershell
pip install -r requirements.txt
```

3) Executar benchmark leve
```powershell
python benchmark_lite.py
```
- O script salva um CSV com os resultados no diretório atual e grava metadados no `mlruns/`.

4) Visualizar resultados
```powershell
mlflow ui --host 127.0.0.1 --port 5001
```
Abra no navegador: http://127.0.0.1:5001

Dicas rápidas
- Se estiver sem GPU: use `benchmark_lite.py`.
- Para liberar espaço: remova `restore_files.zip` ou o diretório `venv/` quando não precisar.
- Para restaurar arquivos removidos: `Expand-Archive -Path restore_files.zip -DestinationPath .`.

Se precisar de instruções passo-a-passo mais detalhadas, consulte o `README.md`.

```markdown
# 🚀 Guia Rápido de Início (Versão Simplificada)

## ⚡ Objetivo

Este repositório foi reduzido para o conjunto mínimo necessário para executar benchmarks leves de modelos Transformer (versão "lite"). O script principal agora é `benchmark_lite.py`.

## ⚙️ O que está neste repositório (essencial)
- `benchmark_lite.py` — Script principal que executa os benchmarks (inference-only para hardware limitado).
- `requirements.txt` — Dependências Python.
- `README.md` — Documentação rápida e detalhes.
- `agricultural_data.py` — Dados de exemplo/sintéticos.
- `mlruns/` — Logs do MLflow (não removido).
- `restore_files.zip` — Backup dos arquivos removidos (caso queira restaurar).

## 💻 Setup Rápido

Abra PowerShell e crie um venv (recomendado):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se tiver GPU NVIDIA/CUDA, instale a versão do PyTorch com CUDA (opcional):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## ▶️ Executar o benchmark (versão lite)
```powershell
python benchmark_lite.py
```
- Saídas: CSV com resultados `benchmark_lite_YYYYMMDD_HHMMSS.csv`.
- Logs de experimento: `mlruns/` (use MLflow UI para visualizar).

## 📊 Visualizar resultados (MLflow)
```powershell
mlflow ui --host 127.0.0.1 --port 5001
```
Acesse: http://127.0.0.1:5001

> Dica: Para abrir o MLflow UI em background (nova janela do PowerShell):
```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command','mlflow ui --host 127.0.0.1 --port 5001'
```

## 💾 Restauração (caso precise dos arquivos removidos)
O arquivo `restore_files.zip` contém os arquivos antigos que foram removidos durante a limpeza. Para restaurar:
```powershell
Expand-Archive -Path restore_files.zip -DestinationPath . -Force
```

## 🧰 Comandos úteis
- Abrir CSV em Python:
```powershell
python -c "import pandas as pd; print(pd.read_csv('benchmark_lite_20251124_114149.csv').head())"
```
- Visualizar imagem do gráfico (se houver):
```powershell
ii .\benchmark_comparison_20251124_121716.png
```

---
**Observação**: `benchmark_lite.py` foi ajustado para usar apenas pipelines de inferência (evita `Trainer`/`accelerate`) para reduzir tempo e dependências.

```markdown
# 🚀 Guia Rápido de Início

## ⚡ Setup Rápido (2 minutos)

### Opção 1: Instalação Automática (Recomendado)

```powershell
# Execute o script de configuração
.\setup.ps1
```

### Opção 2: Instalação Manual

```powershell
# 1. Criar ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Instalar PyTorch (escolha uma opção)

# Com GPU NVIDIA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Sem GPU (CPU):
pip install torch torchvision torchaudio
```

## 🏃 Executar Benchmark

### Para notebooks com MEMÓRIA LIMITADA (< 8GB RAM):

```powershell
python benchmark_lite.py
```

⏱️ **Tempo estimado**: 5-10 minutos  
💾 **Memória necessária**: ~4GB RAM  
📊 **Modelos testados**: DistilBERT + DistilGPT2

### Para notebooks com BOA MEMÓRIA (8GB+ RAM):

```powershell
python benchmark_transformers_sb100.py
```

⏱️ **Tempo estimado**: 20-40 minutos  
💾 **Memória necessária**: ~8GB RAM (16GB recomendado)  
📊 **Modelos testados**: BERT + GPT-2 + BART + DistilBERT + DistilGPT2

## 📊 Visualizar Resultados

### 1. Ver arquivo CSV

Os resultados são salvos automaticamente em:
- `benchmark_results_YYYYMMDD_HHMMSS.csv` (versão completa)
- `benchmark_lite_YYYYMMDD_HHMMSS.csv` (versão lite)

### 2. MLflow UI (Interface Visual)

```powershell
mlflow ui
```

Depois acesse no navegador: **http://localhost:5000**

## 🎯 Qual versão usar?

| Seu Hardware | Script Recomendado | Tempo | Modelos |
|--------------|-------------------|-------|---------|
| RAM < 8GB | `benchmark_lite.py` | ~5-10min | 2 modelos leves |
| RAM 8-16GB | `benchmark_transformers_sb100.py` | ~20-30min | 5 modelos |
| RAM > 16GB + GPU | `benchmark_transformers_sb100.py` | ~15-20min | 5 modelos |

## 🔧 Problemas Comuns

### ❌ "CUDA out of memory"

**Solução**: Use `benchmark_lite.py` ou feche outros programas

### ❌ "Connection error" ao baixar datasets

**Solução**: Verifique sua conexão com a internet. Os datasets são baixados automaticamente do Hugging Face.

### ❌ MLflow não abre

**Solução**: 
```powershell
# Tente porta alternativa
mlflow ui --port 5001
```

## 📈 Interpretando Resultados

### Modelos de Classificação (BERT/DistilBERT)

- **Acurácia**: % de acertos (quanto maior, melhor)
  - Excelente: > 0.90
  - Bom: 0.80 - 0.90
  - Razoável: < 0.80

- **F1-Score**: Equilíbrio entre precisão e recall
  - Excelente: > 0.85
  - Bom: 0.70 - 0.85

### Modelos Generativos (GPT-2/DistilGPT2/BART)

- **Perplexidade**: Incerteza do modelo (quanto menor, melhor)
  - Excelente: < 20
  - Bom: 20 - 40
  - Razoável: > 40

- **BLEU/ROUGE**: Qualidade do texto gerado (quanto maior, melhor)
  - Excelente: > 0.30
  - Bom: 0.15 - 0.30

## 📁 Estrutura dos Arquivos

```
transformer_test/
├── benchmark_transformers_sb100.py  ⭐ Script completo
├── benchmark_lite.py                ⚡ Script leve
├── setup.ps1                        🔧 Instalação automática
├── requirements.txt                 📦 Dependências
├── README.md                        📚 Documentação completa
├── QUICKSTART.md                    🚀 Este arquivo
└── mlruns/                          💾 Resultados MLflow
```

## 💡 Dicas

1. **Primeira vez?** → Use `benchmark_lite.py`
2. **Quer comparar todos os modelos?** → Use `benchmark_transformers_sb100.py`
3. **Notebook lento/travando?** → Feche outros programas e use versão lite
4. **Resultados no MLflow** → Melhor visualização dos experimentos

## 🆘 Precisa de Ajuda?

Consulte o **README.md** para documentação completa, incluindo:
- Explicação detalhada das métricas
- Configuração avançada
- Solução de problemas
- Referências técnicas

---

**Boa sorte com seus benchmarks! 🎉**
