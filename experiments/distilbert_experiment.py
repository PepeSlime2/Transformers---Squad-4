from benchmark_definitivo import benchmark_distilbert_optimized
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_tune', action='store_true', help='Run quick fine-tuning before evaluation')
    args = parser.parse_args()
    benchmark_distilbert_optimized(do_train=args.fine_tune)
