```markdown
# Índice do Projeto

Este documento descreve os arquivos principais e indica como começar com o repositório reduzido.

Arquivos principais
-------------------
- `benchmark_definitivo.py` — script principal para executar benchmarks (versão definitiva, usado pelos experimentos).
- `requirements.txt` — lista de dependências para instalar.
- `agricultural_data.py` — exemplos de dados sintéticos para testes.
- `mlruns/` — diretório onde o MLflow salva os experimentos.
- `restore_files.zip` — backup dos arquivos originais removidos.
- `README.md` — documentação principal.
- `QUICKSTART.md` — guia rápido de uso.

Como começar
-------------
1. Crie e ative um ambiente virtual.
2. Instale as dependências: `pip install -r requirements.txt`.
3. Rode `python benchmark_definitivo.py` para um teste rápido.
4. Inicie o MLflow UI: `mlflow ui --host 127.0.0.1 --port 5001`.

Principais pastas e arquivos gerados
-----------------------------------
- `mlruns/`: resultados dos experimentos (cada run contém parâmetros, métricas e artefatos).
- `benchmark_lite_YYYYMMDD_HHMMSS.csv`: CSV com resultados do benchmark para cada execução.

Suporte e documentação
----------------------
Para problemas e exemplos de comando, consulte `COMMANDS.md` e `README.md`.

# 📑 Índice Completo do Projeto

## Benchmark de Modelos Transformer - Projeto CCD SB100 Squad 4

---

## 📂 Estrutura do Projeto

```
transformer_test/
│
├── 🎯 Scripts Principais
│   ├── benchmark_transformers_sb100.py    # Script completo de benchmark
│   ├── benchmark_lite.py                  # Versão otimizada para hardware limitado (não usado)
│   ├── benchmark_definitivo.py            # Script definitivo de benchmark (usado pelos experimentos)
│   ├── test_agricultural_models.py        # Testes com dados do domínio agrícola
│   └── agricultural_data.py               # Dataset de exemplos agrícolas
│
├── 🧪 Experimentos Individuais
│   └── experiments/
│       ├── run_all_experiments.py          # Executa todos os experimentos
│       ├── transformers_geracao.py         # Experimento: modelos de geração
│       ├── transformers_classificacao.py   # Experimento: modelos de classificação
│       ├── bert_tiny_experiment.py         # BERT Tiny (classificação)
│       ├── chronos_experiment.py           # Chronos (forecasting) - não implementado
│       ├── distilbert_experiment.py        # DistilBERT (classificação)
│       ├── distilgpt2_experiment.py        # DistilGPT-2 (geração)
│       ├── gemini_experiment.py            # Gemma 2 Mini (geração)
│       └── tinyllama_experiment.py         # TinyLlama (geração)
│
├── 📚 Documentação
│   ├── README.md                          # Documentação completa do projeto
│   ├── QUICKSTART.md                      # Guia rápido de início
│   ├── EXECUTIVE_SUMMARY.md               # Resumo executivo
│   ├── COMMANDS.md                        # Comandos úteis
│   └── INDEX.md                           # Este arquivo
│
├── 🔧 Configuração
│   ├── requirements.txt                   # Dependências Python
│   └── setup.ps1                          # Script de instalação automática
│
├── 📊 Resultados (gerados automaticamente)
│   ├── mlruns/                            # Experimentos MLflow
│   ├── results/                           # Checkpoints de modelos
│   ├── results_lite/                      # Checkpoints versão lite
│   └── benchmark_*.csv                    # Resultados em CSV
│
└── 🐍 Ambiente Virtual (criado pelo usuário)
    └── venv/                              # Ambiente virtual Python
```

---

## 📖 Guia de Navegação

### 🎯 Começando

1. **Primeiro acesso?**
   - Leia: [`QUICKSTART.md`](QUICKSTART.md)
   - Execute: `setup.ps1`

2. **Quer entender o projeto?**
   - Leia: [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md)
   - Depois: [`README.md`](README.md)

3. **Precisa de comandos específicos?**
   - Consulte: [`COMMANDS.md`](COMMANDS.md)

### 🚀 Executando

| Objetivo | Arquivo | Hardware Necessário |
|----------|---------|-------------------|
| Teste rápido | `benchmark_definitivo.py` | 4GB+ RAM |
| Benchmark completo | `benchmark_transformers_sb100.py` | 8GB+ RAM |
| Testar com dados agrícolas | `test_agricultural_models.py` | 4GB+ RAM |
| Ver exemplos de dados | `agricultural_data.py` | Qualquer |

### 📊 Analisando Resultados

1. **Arquivos CSV**
   - Formato: `benchmark_results_YYYYMMDD_HHMMSS.csv`
   - Localização: Raiz do projeto
   - Como ver: Excel, pandas, ou qualquer leitor CSV

2. **MLflow UI**
   - Comando: `mlflow ui`
   - URL: http://localhost:5000
   - Dados em: `mlruns/`

---

## 📄 Descrição Detalhada dos Arquivos

### 🎯 Scripts Principais

#### `benchmark_transformers_sb100.py`
- **Tamanho**: ~650 linhas
- **Propósito**: Benchmark completo de 5 modelos Transformer
- **Modelos**: DistilBERT, BERT, DistilGPT2, GPT-2, BART
- **Tempo de execução**: 20-40 minutos
- **Requisitos**: 8GB+ RAM, internet para download de modelos
- **Saída**: CSV + experimentos MLflow
- **Quando usar**: Hardware potente, análise completa

#### `benchmark_lite.py`
- **Tamanho**: ~300 linhas
- **Propósito**: Benchmark otimizado para hardware limitado
- **Modelos**: DistilBERT, DistilGPT2
- **Tempo de execução**: 5-10 minutos
- **Requisitos**: 4GB+ RAM
- **Saída**: CSV + experimentos MLflow
- **Quando usar**: Notebooks com recursos limitados

#### `test_agricultural_models.py`
- **Tamanho**: ~350 linhas
- **Propósito**: Demonstração de uso com dados agrícolas
- **Testes**: 6 diferentes (classificação, QA, geração, etc.)
- **Tempo de execução**: 5-15 minutos
- **Requisitos**: 4GB+ RAM
- **Saída**: Console output
- **Quando usar**: Aprender sobre aplicações práticas

#### `agricultural_data.py`
- **Tamanho**: ~200 linhas
- **Propósito**: Dataset de exemplos do domínio agrícola
- **Conteúdo**: Textos sobre citrus, café, QA, classificação
- **Tempo de execução**: Instantâneo
- **Requisitos**: Nenhum
- **Saída**: Console output (quando executado)
- **Quando usar**: Como fonte de dados para testes

### 📚 Documentação

#### `README.md`
- **Tamanho**: ~500 linhas
- **Conteúdo**: Documentação completa e detalhada
- **Seções**:
  - Descrição do projeto
  - Modelos avaliados
  - Métricas coletadas
  - Instruções de instalação
  - Guia de uso
  - Configuração avançada
  - Solução de problemas
  - Referências técnicas
- **Audiência**: Todos os usuários

#### `QUICKSTART.md`
- **Tamanho**: ~150 linhas
- **Conteúdo**: Guia rápido de início
- **Seções**:
  - Setup rápido (2 minutos)
  - Comandos essenciais
  - Qual versão usar
  - Problemas comuns
  - Interpretação básica
- **Audiência**: Novos usuários

#### `EXECUTIVE_SUMMARY.md`
- **Tamanho**: ~250 linhas
- **Conteúdo**: Resumo executivo do projeto
- **Seções**:
  - Objetivos
  - Estrutura
  - Modelos
  - Métricas
  - Resultados esperados
  - Recomendações
- **Audiência**: Gestores, tomadores de decisão

#### `COMMANDS.md`
- **Tamanho**: ~400 linhas
- **Conteúdo**: Lista de comandos úteis
- **Seções**:
  - Instalação
  - Execução
  - MLflow
  - Verificações
  - Limpeza
  - Debugging
  - Análise
- **Audiência**: Desenvolvedores, power users

#### `INDEX.md`
- **Tamanho**: Este arquivo
- **Conteúdo**: Índice e navegação
- **Propósito**: Mapa do projeto
- **Audiência**: Todos

### 🔧 Configuração

#### `requirements.txt`
- **Tamanho**: ~20 linhas
- **Conteúdo**: Lista de dependências Python
- **Principais pacotes**:
  - torch (PyTorch)
  - transformers (Hugging Face)
  - datasets
  - mlflow
  - evaluate
  - scikit-learn
  - pandas
- **Uso**: `pip install -r requirements.txt`

#### `setup.ps1`
- **Tamanho**: ~80 linhas
- **Conteúdo**: Script PowerShell de instalação
- **Funções**:
  - Cria ambiente virtual
  - Instala dependências
  - Configura PyTorch (CPU/GPU)
  - Verifica instalação
- **Uso**: `.\setup.ps1`

---

## 🔄 Fluxo de Trabalho Recomendado

### Para Iniciantes

```
1. QUICKSTART.md          → Entender básico
2. setup.ps1              → Instalar
3. benchmark_lite.py      → Primeiro teste
4. mlflow ui              → Ver resultados
5. README.md              → Aprofundar
```

### Para Usuários Experientes

```
1. EXECUTIVE_SUMMARY.md           → Overview
2. requirements.txt               → Instalação manual
3. benchmark_transformers_sb100.py → Benchmark completo
4. test_agricultural_models.py    → Testes específicos
5. COMMANDS.md                    → Comandos avançados
```

### Para Pesquisadores

```
1. README.md                      → Metodologia
2. agricultural_data.py           → Explorar dados
3. benchmark_transformers_sb100.py → Modificar parâmetros
4. MLflow                         → Análise detalhada
5. Documentação técnica           → Papers citados
```

---

## 📊 Métricas de Código

### Estatísticas

| Arquivo | Linhas | Comentários | Funções/Classes |
|---------|--------|-------------|-----------------|
| benchmark_transformers_sb100.py | ~650 | ~150 | 15+ métodos |
| benchmark_lite.py | ~300 | ~80 | 4 funções |
| test_agricultural_models.py | ~350 | ~70 | 7 funções |
| agricultural_data.py | ~200 | ~50 | 1 função + dados |
| **Total Python** | **~1500** | **~350** | **30+** |

### Documentação

| Arquivo | Palavras | Páginas equiv. |
|---------|----------|---------------|
| README.md | ~3000 | ~10 |
| QUICKSTART.md | ~800 | ~3 |
| EXECUTIVE_SUMMARY.md | ~1500 | ~5 |
| COMMANDS.md | ~2000 | ~7 |
| **Total Docs** | **~7300** | **~25** |

---

## 🎯 Objetivos de Cada Componente

### Scripts Python

- **benchmark_transformers_sb100.py**: Produzir análise completa e robusta
- **benchmark_lite.py**: Democratizar acesso (hardware limitado)
- **test_agricultural_models.py**: Demonstrar aplicações práticas
- **agricultural_data.py**: Fornecer contexto do domínio

### Documentação

- **README.md**: Ser referência completa
- **QUICKSTART.md**: Reduzir fricção inicial
- **EXECUTIVE_SUMMARY.md**: Comunicar valor e resultados
- **COMMANDS.md**: Ser guia de consulta rápida
- **INDEX.md**: Facilitar navegação

### Configuração

- **requirements.txt**: Garantir reprodutibilidade
- **setup.ps1**: Automatizar configuração

---

## 🔍 Busca Rápida

### Preciso de...

- **Instalar o projeto**: → `QUICKSTART.md` ou `setup.ps1`
- **Entender métricas**: → `README.md` seção "Métricas"
- **Ver exemplos de dados**: → `agricultural_data.py`
- **Comandos específicos**: → `COMMANDS.md`
- **Solucionar erro**: → `README.md` seção "Problemas"
- **Executar teste rápido**: → `benchmark_lite.py`
- **Análise completa**: → `benchmark_transformers_sb100.py`
- **Visualizar resultados**: → `mlflow ui` + CSV
- **Modificar parâmetros**: → Editar scripts Python
- **Referências acadêmicas**: → `README.md` seção "Referências"

---

## 📞 Suporte

### Hierarquia de Documentos

1. **QUICKSTART.md** - Problemas básicos
2. **README.md** - Problemas intermediários
3. **COMMANDS.md** - Comandos específicos
4. **Código-fonte** - Customização avançada

### Ordem de Leitura Sugerida

**Novo no projeto**:
1. QUICKSTART.md
2. EXECUTIVE_SUMMARY.md
3. README.md (parcial)

**Usuário regular**:
1. COMMANDS.md
2. README.md (referência)

**Desenvolvedor/Pesquisador**:
1. README.md (completo)
2. Código-fonte
3. Papers referenciados

---

## 🎓 Recursos Educacionais

### Aprender sobre Transformers
- Código: `benchmark_transformers_sb100.py` (comentado)
- Dados: `agricultural_data.py`
- Testes: `test_agricultural_models.py`
- Teoria: `README.md` seção "Referências"

### Aprender sobre MLflow
- Uso: Todos os scripts de benchmark
- Visualização: `mlflow ui`
- Documentação: Links no README

### Aprender sobre Agricultura
- Dados: `agricultural_data.py`
- Contexto: `EXECUTIVE_SUMMARY.md`

---

## 📅 Versionamento

**Versão**: 1.0  
**Data**: Novembro 2025  
**Projeto**: CCD SB100 – Squad 4  
**Instituição**: Instituto Agronômico de Campinas (IAC)

---

**Este índice é atualizado conforme o projeto evolui**
