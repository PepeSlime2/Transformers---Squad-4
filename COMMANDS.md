## Comandos úteis

Este arquivo contém os comandos mais usados para instalar, rodar e observar resultados do projeto no Windows (PowerShell).

Instalação básica
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Instalar PyTorch com CUDA (opcional)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Executar benchmark leve
```powershell
python benchmark_lite.py
```

Iniciar MLflow
```powershell
mlflow ui --host 127.0.0.1 --port 5001
```
Abra: http://127.0.0.1:5001

Ver arquivos CSV gerados
```powershell
Get-ChildItem -Filter "benchmark_*.csv" | Sort-Object LastWriteTime -Descending
```

Remover resultados antigos
```powershell
Remove-Item benchmark_*.csv -Force
```

Restaurar backup (se necessário)
```powershell
Expand-Archive -Path restore_files.zip -DestinationPath . -Force
```

Dicas rápidas
- Se o MLflow não abrir na porta padrão, tente outra porta: `mlflow ui --port 5002`.
- Se faltar memória GPU, use a versão `benchmark_lite.py`.
- Se erro de importação ocorrer, rode `pip install -r requirements.txt`.

Se precisar de comandos específicos para Linux ou para execução com GPU, me avise.

```markdown
# Comandos úteis - Benchmark Transformers (versão simplificada)

Este arquivo reúne os comandos mais usados para instalar, executar e depurar o projeto no Windows (PowerShell).

1) Criar e ativar ambiente virtual
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2) Instalar dependências
```powershell
pip install -r requirements.txt
```

3) Instalar PyTorch com suporte a CUDA (opcional)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4) Executar o benchmark (versão leve)
```powershell
python benchmark_lite.py
```

5) Iniciar a interface de visualização (MLflow)
```powershell
mlflow ui --host 127.0.0.1 --port 5001
```
Abra no navegador: http://127.0.0.1:5001

6) Comandos úteis para inspeção rápida
```powershell
# Ver a versão do Python
python --version

# Verificação rápida do PyTorch e CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Listar CSVs gerados (ordenados por data)
Get-ChildItem -Filter "benchmark_*.csv" | Sort-Object LastWriteTime -Descending

# Abrir CSV no Excel (ou no aplicativo associado)
Start-Process "benchmark_lite_*.csv"
```

7) Limpeza (remover resultados antigos)
```powershell
Remove-Item benchmark_*.csv -Force
Remove-Item -Recurse -Force results results_lite
```

8) Restaurar arquivos removidos (backup)
```powershell
Expand-Archive -Path restore_files.zip -DestinationPath . -Force
```

9) Parar MLflow (feche a janela onde o servidor está rodando) ou mate o processo
```powershell
Get-Process python | Where-Object {$_.Path -like '*mlflow*' } | Stop-Process
```

10) Dicas rápidas
- Se o MLflow não abrir, tente outra porta: `mlflow ui --port 5002`
- Se faltar memória GPU, use a versão `benchmark_lite.py`.
- Se ocorrer erro de importação, rode: `pip install -r requirements.txt`

Se precisar de mais comandos, posso incluir instruções específicas para Linux ou para execução com GPU.
```
# Comandos Úteis - Projeto Benchmark Transformers SB100

## 🔧 Instalação e Configuração

### Criar ambiente virtual
```powershell
python -m venv venv
```

### Ativar ambiente virtual
```powershell
# PowerShell
.\venv\Scripts\Activate.ps1

# CMD
.\venv\Scripts\activate.bat
```

### Desativar ambiente virtual
```powershell
deactivate
```

### Instalar dependências
```powershell
# Todas as dependências
pip install -r requirements.txt

# PyTorch com GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch sem GPU (CPU)
pip install torch torchvision torchaudio

# Atualizar pip
python -m pip install --upgrade pip
```

## 🚀 Execução dos Scripts

### Benchmark completo
```powershell
python benchmark_transformers_sb100.py
```

### Benchmark leve (recomendado)
```powershell
python benchmark_lite.py
```

### Testes com dados agrícolas
```powershell
python test_agricultural_models.py
```

### Visualizar dados agrícolas
```powershell
python agricultural_data.py
```

## 📊 MLflow

### Iniciar interface MLflow
```powershell
mlflow ui
```

### Iniciar em porta específica
```powershell
mlflow ui --port 5001
```

### Iniciar em host específico
```powershell
mlflow ui --host 0.0.0.0 --port 5000
```

### Ver experimentos específicos
```powershell
mlflow experiments list
```

### Limpar cache do MLflow
```powershell
Remove-Item -Recurse -Force mlruns
```

## 🐍 Verificações Python/PyTorch

### Verificar versão Python
```powershell
python --version
```

### Verificar instalação PyTorch
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Verificar suporte CUDA
```powershell
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

### Verificar GPU
```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Sem GPU')"
```

### Verificar memória GPU
```powershell
python -c "import torch; print(f'Memória total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB' if torch.cuda.is_available() else 'Sem GPU')"
```

### Verificar todas as bibliotecas
```powershell
python -c "import torch, transformers, mlflow, datasets, evaluate; print('✅ Todas as bibliotecas instaladas!')"
```

## 📦 Gerenciamento de Pacotes

### Listar pacotes instalados
```powershell
pip list
```

### Verificar versão específica
```powershell
pip show transformers
pip show torch
pip show mlflow
```

### Atualizar pacote específico
```powershell
pip install --upgrade transformers
```

### Desinstalar pacote
```powershell
pip uninstall torch
```

### Criar requirements.txt do ambiente atual
```powershell
pip freeze > requirements_frozen.txt
```

## 🧹 Limpeza e Manutenção

### Limpar cache Python
```powershell
# PowerShell
Get-ChildItem -Path . -Include __pycache__,*.pyc -Recurse | Remove-Item -Force -Recurse
```

### Limpar resultados antigos
```powershell
Remove-Item -Path ".\results" -Recurse -Force
Remove-Item -Path ".\results_lite" -Recurse -Force
```

### Limpar cache do Hugging Face
```powershell
$env:HF_HOME
Remove-Item -Path "$env:HF_HOME\hub" -Recurse -Force
```

### Limpar CSVs antigos
```powershell
Remove-Item benchmark_*.csv
```

## 🔍 Monitoramento

### Ver processos Python
```powershell
Get-Process python
```

### Ver uso de memória
```powershell
Get-Process python | Select-Object ProcessName, @{Name='Memory(MB)';Expression={$_.WorkingSet / 1MB}}
```

### Verificar espaço em disco
```powershell
Get-PSDrive C | Select-Object Used,Free
```

### Ver porta ocupada
```powershell
netstat -ano | findstr :5000
```

## 📊 Análise de Resultados

### Ver últimos CSVs gerados
```powershell
Get-ChildItem -Filter "benchmark_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### Abrir CSV no Excel
```powershell
Start-Process "benchmark_results_*.csv"
```

### Ver conteúdo do CSV
```powershell
Import-Csv "benchmark_results_20241112_*.csv" | Format-Table
```

### Comparar dois CSVs
```powershell
$csv1 = Import-Csv "benchmark_lite_20241112_100000.csv"
$csv2 = Import-Csv "benchmark_lite_20241112_110000.csv"
Compare-Object $csv1 $csv2 -Property model_name, accuracy
```

## 🐛 Debugging

### Executar com verbose
```powershell
python -v benchmark_lite.py
```

### Ver importações Python
```powershell
python -v -c "import transformers"
```

### Testar importações
```powershell
python -c "import sys; print('\n'.join(sys.path))"
```

### Ver variáveis de ambiente
```powershell
Get-ChildItem Env: | Where-Object {$_.Name -like "*PYTHON*" -or $_.Name -like "*CUDA*"}
```

## 🔐 Configuração Avançada

### Configurar cache do Hugging Face
```powershell
$env:HF_HOME = "C:\Users\Pepe\.cache\huggingface"
$env:TRANSFORMERS_CACHE = "C:\Users\Pepe\.cache\transformers"
```

### Desabilitar telemetria
```powershell
$env:TRANSFORMERS_OFFLINE = "1"
```

### Usar GPU específica (se houver múltiplas)
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
```

### Limitar uso de CPU
```powershell
$env:OMP_NUM_THREADS = "4"
```

## 📝 Logs e Outputs

### Salvar output em arquivo
```powershell
python benchmark_lite.py > output.log 2>&1
```

### Ver log em tempo real
```powershell
python benchmark_lite.py | Tee-Object -FilePath output.log
```

### Contar linhas de código
```powershell
(Get-Content benchmark_transformers_sb100.py).Count
```

## 🔄 Git (opcional)

### Inicializar repositório
```powershell
git init
```

### Adicionar arquivos
```powershell
git add *.py *.txt *.md
```

### Criar .gitignore
```powershell
@"
venv/
__pycache__/
*.pyc
mlruns/
results/
results_lite/
*.csv
*.log
.env
"@ | Out-File -FilePath .gitignore -Encoding UTF8
```

### Commit
```powershell
git commit -m "Initial commit - Benchmark Transformers SB100"
```

## 📊 Exemplos de One-Liners Úteis

### Ver tamanho total do projeto
```powershell
(Get-ChildItem -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
```

### Contar arquivos Python
```powershell
(Get-ChildItem -Filter "*.py" -Recurse).Count
```

### Listar modelos baixados
```powershell
Get-ChildItem "$env:HF_HOME\hub" -Directory
```

### Ver último benchmark executado
```powershell
Get-ChildItem "benchmark_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

## 💡 Dicas de Performance

### Usar menos cores CPU
```powershell
$env:MKL_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
```

### Pré-download de modelos
```powershell
python -c "from transformers import AutoModel; AutoModel.from_pretrained('distilbert-base-uncased')"
```

### Verificar CUDA toolkit
```powershell
nvcc --version
```

## 🆘 Troubleshooting

### Reinstalar ambiente do zero
```powershell
deactivate
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Forçar reinstalação
```powershell
pip install --force-reinstall --no-cache-dir torch transformers
```

### Verificar conflitos de pacotes
```powershell
pip check
```

---

**Nota**: Estes comandos são para PowerShell no Windows. Para outros shells ou sistemas operacionais, adapte conforme necessário.
