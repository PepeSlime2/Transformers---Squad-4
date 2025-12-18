#!/usr/bin/env python
"""
Experimento: Transformers_Classificação
Executa os modelos de classificação em sequência.
Modelos incluídos: DistilBERT, BERT Tiny
"""
import subprocess
import sys
import os
from time import sleep

EXPERIMENT_SCRIPTS = [
    'distilbert_experiment.py',
    'bert_tiny_experiment.py',
]

SLEEP_BETWEEN = 5  # segundos

if __name__ == '__main__':
    # Permite passar --fine_tune para os scripts de classificação
    fine_tune_flag = '--fine_tune' if '--fine_tune' in sys.argv else ''

    # Run experiments from repository root so imports like `benchmark_lite` work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cwd = repo_root
    for script in EXPERIMENT_SCRIPTS:
        script_path = os.path.join(cwd, 'experiments', script)
        cmd = [sys.executable, script_path]
        # Passa o flag --fine_tune para scripts de classificação
        if fine_tune_flag and ('distilbert' in script or 'bert_tiny' in script):
            cmd.append('--fine_tune')
        print(f"\n Iniciando: {script} -> comando: {' '.join(cmd)}")
        # ensure Python sees project root in sys.path and run from repo root
        env = os.environ.copy()
        env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
        env['MLFLOW_EXPERIMENT_NAME'] = 'Transformers_Classificação'
        proc = subprocess.Popen(cmd, cwd=cwd, env=env)
        ret = proc.wait()
        print(f" Script {script} finalizou com código: {ret}")
        if ret != 0:
            print(f" O script {script} retornou erro (code={ret}). Confira logs e mlflow/outputs.")
        print(f" Aguardando {SLEEP_BETWEEN}s antes de iniciar o próximo...")
        sleep(SLEEP_BETWEEN)

    print('\n Experimento Transformers_Classificação concluído. Veja os runs no MLflow ou CSVs gerados.')