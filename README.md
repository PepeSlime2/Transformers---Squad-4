# Benchmark de Transformers - Squad 4 CCD SB100

Projeto de benchmark de modelos Transformer otimizado para hardware limitado.

## Modelos Testados

- **Classificação**: DistilBERT, BERT Tiny, RoBERTa Mini(não testado)
- **Geração**: DistilGPT2, TinyLLaMA, Gemma 2 Mini

## Começando

```bash
git clone https://github.com/PepeSlime2/Transformers---Squad-4.git
cd Transformers---Squad-4
pip install -r requirements.txt
python src/benchmark.py