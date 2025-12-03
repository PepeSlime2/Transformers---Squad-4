from benchmark_lite import benchmark_bert_tiny
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_tune', action='store_true', help='Run quick fine-tuning before evaluation')
    args = parser.parse_args()
    benchmark_bert_tiny(do_train=args.fine_tune)
