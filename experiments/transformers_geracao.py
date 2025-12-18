#!/usr/bin/env python
"""
Experimento: Transformers_Geração
Executa os modelos de geração de texto em sequência.
Modelos incluídos: DistilGPT-2, Gemini, TinyLlama
"""
import subprocess
import sys
import os
from time import sleep

EXPERIMENT_SCRIPTS = [
    'distilgpt2_experiment.py',
    'gemini_experiment.py',
    'tinyllama_experiment.py',
]

SLEEP_BETWEEN = 5  # segundos

if __name__ == '__main__':
    # Run experiments from repository root so imports like `benchmark_lite` work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cwd = repo_root
    for script in EXPERIMENT_SCRIPTS:
        script_path = os.path.join(cwd, 'experiments', script)
        cmd = [sys.executable, script_path]
        print(f"\ Iniciando: {script} -> comando: {' '.join(cmd)}")
        # ensure Python sees project root in sys.path and run from repo root
        env = os.environ.copy()
        env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
        env['MLFLOW_EXPERIMENT_NAME'] = 'Transformers_Geração'
        proc = subprocess.Popen(cmd, cwd=cwd, env=env)
        ret = proc.wait()
        print(f" Script {script} finalizou com código: {ret}")
        if ret != 0:
            print(f" O script {script} retornou erro (code={ret}). Confira logs e mlflow/outputs.")
        print(f"Aguardando {SLEEP_BETWEEN}s antes de iniciar o próximo...")
        sleep(SLEEP_BETWEEN)

    print('\n Experimento Transformers_Geração concluído. Veja os runs no MLflow ou CSVs gerados.')