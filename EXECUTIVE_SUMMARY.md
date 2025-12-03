```markdown
# Resumo Executivo - Benchmark Transformers (versão simplificada)

Objetivo
---------
Avaliar o desempenho de modelos Transformer em tarefas de classificação e geração, usando conjuntos de dados reduzidos para facilitar a execução em equipamentos com recursos limitados.

O que este repositório contém
--------------------------------
- `benchmark_lite.py`: script principal para avaliação rápida (inference-only).
- `benchmark_lite_*.csv`: resultados consolidados em CSV.
- `mlruns/`: registros dos experimentos no MLflow.
- `restore_files.zip`: backup dos arquivos originais, caso precise restaurar algo.

Modelos utilizados nesta versão
------------------------------
- DistilBERT (classificação)
- DistilGPT2 (geração)

Principais métricas
-------------------
- Acurácia, Precisão, Recall, F1 (classificação)
- Perplexidade, BLEU, ROUGE (geração) — quando aplicável
- Tempo de inferência e tempo de geração
- Tamanho do modelo (MB) e número de parâmetros

Resumo dos resultados (exemplo)
-------------------------------
- DistilBERT: rápido, leve, accuracy baixa se não for finetuned para a tarefa específica.
- DistilGPT2: gera textos coerentes em amostras curtas; tempo de geração aceitável em CPU.

Recomendações
-------------
1. Para a maioria das avaliações rápidas, use `benchmark_lite.py`.
2. Caso precise de avaliação mais completa ou fine-tuning, execute os scripts de versão completa a partir do backup.
3. Use MLflow para comparar runs e consultar métricas detalhadas.

Próximos passos sugeridos
-------------------------
1. Subir amostras reais do domínio agrícola e reavaliar os modelos.
2. Fazer fine-tuning em modelos leves (se for necessário melhorar accuracy).
3. Automatizar a coleta de métricas para rodar comparações periódicas.

Versionamento
-------------
Versão atual: 1.0 (Novembro 2025), versão simplificada focada em inferência.

# 📊 Resumo Executivo - Benchmark Transformers SB100

## Projeto CCD SB100 – Squad 4
**Instituto Agronômico de Campinas (IAC)**

---

## 🎯 Objetivo

Realizar benchmarks comparativos de modelos Transformer (BERT, GPT-2, BART e variantes) para avaliar seu desempenho em tarefas relacionadas ao domínio agrícola, especificamente textos do Boletim 100 do IAC.

## 📁 Arquivos do Projeto

| Arquivo | Descrição | Quando Usar |
|---------|-----------|-------------|
| `benchmark_transformers_sb100.py` | ⭐ Script completo de benchmark | Hardware com 8GB+ RAM |
| `benchmark_lite.py` | ⚡ Versão otimizada | Hardware com < 8GB RAM |
| `test_agricultural_models.py` | 🧪 Testes com dados agrícolas | Demonstração de uso |
| `agricultural_data.py` | 📚 Dataset de exemplos | Fonte de dados |
| `setup.ps1` | 🔧 Instalação automática | Primeira vez |
| `requirements.txt` | 📦 Dependências | Instalação manual |
| `README.md` | 📖 Documentação completa | Referência detalhada |
| `QUICKSTART.md` | 🚀 Guia rápido | Início rápido |

## 🤖 Modelos Avaliados

### Encoder (Classificação)
- **DistilBERT** - 67M parâmetros (leve e rápido)
- **BERT** - 110M parâmetros (mais preciso)

### Decoder/Seq2Seq (Geração)
- **DistilGPT-2** - 82M parâmetros (leve e rápido)
- **GPT-2** - 124M parâmetros (mais criativo)
- **BART** - 140M parâmetros (seq2seq)

### Nota sobre Modelos Grandes
- **LLaMA** e **DeepSeek** requerem 8GB+ VRAM
- Para hardware limitado, use modelos Distil* como alternativas válidas

## 📊 Métricas Coletadas

### ✅ Precisão
| Métrica | Tipo de Modelo | Interpretação |
|---------|----------------|---------------|
| Acurácia | Classificação | % de previsões corretas |
| Precisão | Classificação | % de positivos corretos |
| Recall | Classificação | % de positivos encontrados |
| F1-Score | Classificação | Média harmônica P/R |
| BLEU | Geração | Qualidade do texto gerado |
| ROUGE | Geração | Sobreposição de n-gramas |
| Perplexidade | Geração | Incerteza do modelo |

### ⚡ Eficiência
- Tempo de treinamento (segundos)
- Tempo de inferência (segundos)
- Amostras processadas por segundo

### 💾 Recursos
- Número de parâmetros
- Tamanho do modelo (MB)
- Memória GPU (VRAM)

## 🚀 Como Executar

### Instalação (5 minutos)
```powershell
# Automática
.\setup.ps1

# Manual
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Execução

**Opção 1: Versão Leve (Recomendado para maioria)**
```powershell
python benchmark_lite.py
```
- ⏱️ Tempo: 5-10 minutos
- 💾 RAM necessária: ~4GB
- 📊 Testa: DistilBERT + DistilGPT2

**Opção 2: Versão Completa**
```powershell
python benchmark_transformers_sb100.py
```
- ⏱️ Tempo: 20-40 minutos
- 💾 RAM necessária: 8GB+ (16GB recomendado)
- 📊 Testa: 5 modelos diferentes

**Opção 3: Testes com Dados Agrícolas**
```powershell
python test_agricultural_models.py
```
- ⏱️ Tempo: 5-15 minutos
- Demonstra uso prático dos modelos

### Visualização de Resultados
```powershell
mlflow ui
# Acesse: http://localhost:5000
```

## 📈 Resultados Esperados

### Modelos de Classificação (AG News Dataset)

| Modelo | Acurácia | F1-Score | Tempo Treino | Parâmetros |
|--------|----------|----------|--------------|------------|
| DistilBERT | ~0.89 | ~0.88 | ~45s | 67M |
| BERT | ~0.91 | ~0.90 | ~90s | 110M |

### Modelos Generativos (WikiText Dataset)

| Modelo | Perplexidade | BLEU | ROUGE-L | Tempo Inf. | Parâmetros |
|--------|--------------|------|---------|------------|------------|
| DistilGPT2 | ~32 | ~0.15 | ~0.22 | ~12s | 82M |
| GPT-2 | ~28 | ~0.18 | ~0.25 | ~23s | 124M |
| BART | ~25 | ~0.21 | ~0.29 | ~31s | 140M |

**Nota**: Valores aproximados, variam conforme hardware e configuração.

## 🔍 Interpretação dos Resultados

### Para Classificação
- **Acurácia > 0.85**: Excelente
- **F1-Score > 0.80**: Bom equilíbrio

### Para Geração
- **Perplexidade < 30**: Excelente
- **BLEU > 0.20**: Bom
- **ROUGE-L > 0.25**: Bom

## 💡 Recomendações

### Para Hardware Limitado (< 8GB RAM)
1. ✅ Use `benchmark_lite.py`
2. ✅ Teste apenas DistilBERT e DistilGPT2
3. ✅ Reduza `sample_size` no código se necessário

### Para Hardware Médio (8-16GB RAM)
1. ✅ Use `benchmark_transformers_sb100.py`
2. ✅ Feche outros programas durante execução
3. ✅ Monitore uso de memória

### Para Hardware Avançado (16GB+ RAM, GPU)
1. ✅ Use `benchmark_transformers_sb100.py`
2. ✅ Aumente `sample_size` para melhor avaliação
3. ✅ Considere testar modelos maiores (LLaMA, etc.)

## 🎓 Aplicações no Domínio Agrícola

### Casos de Uso
1. **Classificação de Documentos** - Categorizar relatórios técnicos
2. **Extração de Informações** - Identificar práticas recomendadas
3. **Geração de Textos** - Criar resumos de boletins
4. **Question Answering** - Sistema de perguntas sobre culturas
5. **Análise de Sentimento** - Avaliar percepção sobre tecnologias

### Datasets Específicos
- Textos sobre Citrus (5 amostras)
- Textos sobre Café (5 amostras)
- Pares de Q&A (5 exemplos)
- Categorias agrícolas

## 📊 Estrutura do MLflow

```
mlruns/
└── Benchmark_Transformers_SB100/
    ├── Run 1: DistilBERT_classification
    ├── Run 2: BERT_classification
    ├── Run 3: DistilGPT2_generation
    ├── Run 4: GPT2_generation
    └── Run 5: BART_generation
```

Cada run contém:
- **Parâmetros**: modelo, hiperparâmetros, device
- **Métricas**: todas as métricas de desempenho
- **Artefatos**: logs, checkpoints

## ⚠️ Problemas Comuns e Soluções

| Problema | Solução |
|----------|---------|
| CUDA out of memory | Use `benchmark_lite.py` |
| Download lento | Verifique conexão internet |
| MLflow não abre | Use `mlflow ui --port 5001` |
| Importações falhando | Execute `pip install -r requirements.txt` |

## 📚 Referências Técnicas

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [BART Paper](https://arxiv.org/abs/1910.13461)

## 📝 Checklist de Uso

- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas
- [ ] PyTorch instalado (GPU ou CPU)
- [ ] Script de benchmark executado
- [ ] Resultados salvos em CSV
- [ ] MLflow UI acessado
- [ ] Métricas analisadas
- [ ] Comparações realizadas

## 🎯 Próximos Passos

1. **Coletar dados reais** do Boletim 100 do IAC
2. **Fine-tuning** dos modelos com dados agrícolas
3. **Validação** com especialistas do domínio
4. **Deployment** de modelo(s) selecionado(s)
5. **Integração** em sistema de consulta

## 👥 Créditos

**Projeto**: CCD SB100 – Squad 4  
**Instituição**: Instituto Agronômico de Campinas (IAC)  
**Data**: Novembro 2025

---

**Para mais informações, consulte README.md ou QUICKSTART.md**
