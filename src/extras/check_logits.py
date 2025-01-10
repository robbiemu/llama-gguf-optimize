import h5py
import numpy as np
import argparse


def check_logits_completion(baseline_file, target_file, chunks):
    """Check completion status of logits files for specified chunks"""
    print("\nChecking completion status...")
    
    with h5py.File(baseline_file, 'r') as f:
        baseline_processed = f['processed_chunks'][:]
        baseline_total = f.attrs.get('total_chunks', len(baseline_processed))
        print(f"\nBaseline file ({baseline_file}):")
        print(f"Total chunks: {baseline_total}")
        print(f"Total processed chunks: {np.sum(baseline_processed)}")
        
        for chunk in chunks:
            print(f"\nChunk {chunk} baseline status:")
            print(f"Processed flag: {'Processed' if baseline_processed[chunk] else 'Not processed'}")
            if 'logits' in f:
                chunk_data = f['logits'][chunk]
                has_data = np.any(chunk_data != 0)
                print(f"Logits data present: {has_data}")
    
    with h5py.File(target_file, 'r') as f:
        target_processed = f['processed_chunks'][:]
        target_total = f.attrs.get('total_chunks', len(target_processed))
        print(f"\nTarget file ({target_file}):")
        print(f"Total chunks: {target_total}")
        print(f"Total processed chunks: {np.sum(target_processed)}")
        
        for chunk in chunks:
            print(f"\nChunk {chunk} target status:")
            print(f"Processed flag: {'Processed' if target_processed[chunk] else 'Not processed'}")
            if 'logits' in f:
                chunk_data = f['logits'][chunk]
                has_data = np.any(chunk_data != 0)
                print(f"Logits data present: {has_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check completion status of logits files")
    parser.add_argument('--baseline', default="baseline_logits.h5", help="Baseline logits file")
    parser.add_argument('--target', default="target_logits.h5", help="Target logits file")
    parser.add_argument('--chunks', type=int, nargs='+', required=True, 
                        help="Space-separated list of chunk numbers to check")
    
    args = parser.parse_args()
    check_logits_completion(args.baseline, args.target, args.chunks)