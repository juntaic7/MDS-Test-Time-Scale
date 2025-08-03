#!/usr/bin/env python3
"""
Evaluation script for ROUGE and BERTScore metrics.
Usage: python eval.py <file_path>

Example: python eval.py summaries/openasp/model_name/nc_2.jsonl
"""

import argparse
import json
import sys
import os
from pathlib import Path
from evaluate import load
from utils import read_jsonl
import torch
import numpy as np

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def load_dataset_references(dataset_name, indices, use_local_gold=False):
    """Load reference summaries for the given dataset and indices."""
    if use_local_gold:
        # Load from local gold summary files
        gold_file = f"summaries/{dataset_name}/gold_summaries.jsonl"
        if not Path(gold_file).exists():
            raise FileNotFoundError(f"Gold summaries file not found: {gold_file}")
        
        gold_data = read_jsonl(gold_file)
        # Create a mapping from id to summary for quick lookup
        gold_summaries = {int(item['id']): item['result'] for item in gold_data}
        
        # Get references for the specified indices
        references = []
        for idx in indices:
            if idx not in gold_summaries:
                raise ValueError(f"Index {idx} not found in gold summaries")
            references.append(gold_summaries[idx])
        
    else:
        # Load from original datasets
        if dataset_name == "multinews":
            from datasets import load_dataset
            test = load_dataset("alexfabbri/multi_news", split="test")
            references = [test['summary'][i] for i in indices]
        elif dataset_name == "openasp":
            with open("datasets/openasp-v1/test.jsonl") as f:
                test = [json.loads(line) for line in f]
            references = [" ".join(test[i]['summary_text']) for i in indices]
        else:
            raise ValueError(f"Unrecognized dataset: {dataset_name}")
    
    return references


def filter_outlier_summaries(data, outlier_multiplier=5.0):
    """Filter out summaries that are significantly longer than others (likely erroneous)."""
    # Calculate summary lengths
    lengths = [len(item['result']) for item in data]
    
    # Calculate statistics
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    
    print(f"Summary length statistics:", file=sys.stderr)
    print(f"  Mean: {mean_length:.1f} chars", file=sys.stderr)
    print(f"  Median: {median_length:.1f} chars", file=sys.stderr)
    print(f"  Min: {min(lengths)} chars", file=sys.stderr)
    print(f"  Max: {max(lengths)} chars", file=sys.stderr)
    
    # Define outlier threshold (multiplier * mean)
    upper_limit = outlier_multiplier * mean_length
    print(f"  Outlier threshold (>{outlier_multiplier}x mean): {upper_limit:.1f} chars", file=sys.stderr)
    
    # Filter data
    filtered_data = []
    outliers = []
    
    for item in data:
        length = len(item['result'])
        if length <= upper_limit:
            filtered_data.append(item)
        else:
            outliers.append((item['id'], length))
            print(f"  Filtered outlier: ID {item['id']} ({length} chars, {length/mean_length:.1f}x mean)", file=sys.stderr)
    
    print(f"Filtered {len(outliers)} outliers, keeping {len(filtered_data)}/{len(data)} summaries", file=sys.stderr)
    
    return filtered_data, outliers


def extract_dataset_from_path(file_path):
    """Extract dataset name from file path like summaries/openasp/model/file.jsonl"""
    path_parts = Path(file_path).parts
    if len(path_parts) >= 2 and path_parts[0] == "summaries":
        return path_parts[1]
    else:
        raise ValueError(f"Cannot extract dataset from path: {file_path}. Expected format: summaries/dataset/...")


def load_raw_predictions(file_path):
    """Load predictions from raw GPT output format."""
    data = []
    for result in read_jsonl(file_path):
        try:
            # Extract custom_id to get the index
            custom_id = result['custom_id']
            idx = int(custom_id.split('-')[1])  # Extract from "request-X"
            
            # Navigate through the nested structure to get the content
            content = result['response']['body']['choices'][0]['message']['content']
            
            data.append({
                'id': str(idx),
                'result': content
            })
        except (KeyError, ValueError, IndexError) as e:
            print(f"Warning: Failed to parse raw result {custom_id if 'custom_id' in result else 'unknown'}: {e}", file=sys.stderr)
            continue
    
    return data


def evaluate_file(file_path, dataset_name, use_local_gold=False, filter_outliers=True, raw_format=False):
    """Evaluate using ROUGE and BERTScore metrics with multi-GPU support."""
    # Load predictions based on format
    if raw_format:
        print(f"Loading raw GPT output format from {file_path}...", file=sys.stderr)
        data = load_raw_predictions(file_path)
    else:
        data = read_jsonl(file_path)
    
    original_count = len(data)
    
    # Filter outliers if requested
    if filter_outliers:
        data, outliers = filter_outlier_summaries(data)
        if outliers:
            print(f"Warning: {len(outliers)} outliers filtered from evaluation", file=sys.stderr)
    
    # Extract indices and predictions from filtered data
    indices = [int(line['id']) for line in data]
    predictions = [line['result'] for line in data]
    references = load_dataset_references(dataset_name, indices, use_local_gold)
    
    # Load metrics and compute scores
    rouge_metric = load("rouge")
    
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    
    # BERTScore with multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for BERTScore evaluation", file=sys.stderr)
        
        def process_gpu_batch(gpu_id, batch_predictions, batch_references, batch_start_idx):
            """Process a batch of predictions on a specific GPU"""
            device = f"cuda:{gpu_id}"
            micro_batch_size = 16
            
            print(f"GPU {gpu_id}: processing {len(batch_predictions)} samples", file=sys.stderr)
            
            gpu_precision = []
            gpu_recall = []
            gpu_f1 = []
            
            # Load BERTScore metric for this GPU
            with torch.cuda.device(gpu_id):
                bertscore_metric = load("bertscore")
                
                # Process in micro-batches
                for i in range(0, len(batch_predictions), micro_batch_size):
                    end_i = min(i + micro_batch_size, len(batch_predictions))
                    micro_preds = batch_predictions[i:end_i]
                    micro_refs = batch_references[i:end_i]
                    
                    print(f"  GPU {gpu_id}: micro-batch {batch_start_idx + i}-{batch_start_idx + end_i - 1}", file=sys.stderr)
                    
                    micro_results = bertscore_metric.compute(
                        predictions=micro_preds,
                        references=micro_refs,
                        lang="en",
                        model_type="microsoft/deberta-xlarge-mnli",
                        device=device
                    )
                    
                    gpu_precision.extend(micro_results["precision"])
                    gpu_recall.extend(micro_results["recall"])
                    gpu_f1.extend(micro_results["f1"])
                    
                    torch.cuda.empty_cache()
                
                # Cleanup
                del bertscore_metric
                torch.cuda.empty_cache()
                
            print(f"GPU {gpu_id}: completed {len(gpu_precision)} samples", file=sys.stderr)
            return gpu_precision, gpu_recall, gpu_f1
        
        # Split data across GPUs
        n_gpus = torch.cuda.device_count()
        batch_size = len(predictions) // n_gpus
        
        # Prepare GPU batches
        gpu_tasks = []
        for gpu_id in range(n_gpus):
            start_idx = gpu_id * batch_size
            end_idx = (gpu_id + 1) * batch_size if gpu_id < n_gpus - 1 else len(predictions)
            
            if start_idx < len(predictions):
                batch_preds = predictions[start_idx:end_idx]
                batch_refs = references[start_idx:end_idx]
                gpu_tasks.append((gpu_id, batch_preds, batch_refs, start_idx))
        
        # Process batches sequentially on different GPUs (safer than threading with CUDA)
        all_precision = []
        all_recall = []
        all_f1 = []
        
        for gpu_id, batch_preds, batch_refs, start_idx in gpu_tasks:
            try:
                precision, recall, f1 = process_gpu_batch(gpu_id, batch_preds, batch_refs, start_idx)
                all_precision.extend(precision)
                all_recall.extend(recall)
                all_f1.extend(f1)
            except Exception as exc:
                print(f'GPU {gpu_id} generated an exception: {exc}', file=sys.stderr)
        
        bertscore_results = {
            "precision": all_precision,
            "recall": all_recall,
            "f1": all_f1
        }
    elif torch.cuda.is_available():
        # Single GPU fallback
        micro_batch_size = 32
        print(f"Using single GPU for BERTScore evaluation with micro-batches of {micro_batch_size}", file=sys.stderr)
        
        all_precision = []
        all_recall = []
        all_f1 = []
        
        bertscore_metric = load("bertscore")
        
        for micro_start in range(0, len(predictions), micro_batch_size):
            micro_end = min(micro_start + micro_batch_size, len(predictions))
            micro_predictions = predictions[micro_start:micro_end]
            micro_references = references[micro_start:micro_end]
            
            print(f"  Processing micro-batch {micro_start}-{micro_end-1} ({len(micro_predictions)} samples)", file=sys.stderr)
            
            micro_results = bertscore_metric.compute(
                predictions=micro_predictions,
                references=micro_references,
                lang="en",
                model_type="microsoft/deberta-xlarge-mnli",
                device="cuda:0"
            )
            
            all_precision.extend(micro_results["precision"])
            all_recall.extend(micro_results["recall"])
            all_f1.extend(micro_results["f1"])
            
            torch.cuda.empty_cache()
        
        del bertscore_metric
        torch.cuda.empty_cache()
        
        bertscore_results = {
            "precision": all_precision,
            "recall": all_recall,
            "f1": all_f1
        }
    else:
        # Single GPU or CPU fallback
        bertscore_metric = load("bertscore")
        bertscore_results = bertscore_metric.compute(
            predictions=predictions, 
            references=references, 
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )
    
    # Clear CUDA cache to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Calculate average word count of predictions
    word_counts = [len(pred.split()) for pred in predictions]
    avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
    
    result = {
        "file": str(file_path),
        "total_samples": original_count,
        "evaluated_samples": len(predictions),
        "filtered_outliers": original_count - len(predictions) if filter_outliers else 0,
        "avg_word_count": round(avg_word_count, 1),
        "rouge": {
            "rouge1": rouge_results.get("rouge1", 0.0),
            "rouge2": rouge_results.get("rouge2", 0.0),
            "rougeL": rouge_results.get("rougeL", 0.0),
            "rougeLsum": rouge_results.get("rougeLsum", 0.0)
        },
        "bertscore": {
            "precision": sum(bertscore_results["precision"]) / len(bertscore_results["precision"]),
            "recall": sum(bertscore_results["recall"]) / len(bertscore_results["recall"]),
            "f1": sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
        }
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate summaries with ROUGE and BERTScore")
    parser.add_argument("file_path", help="Path to summary file (e.g., summaries/openasp/model_name/nc_2.jsonl)")
    parser.add_argument("--output", "-o", help="Optional output file to save results")
    parser.add_argument("--gold-local", action="store_true",
                        help="Use local gold summaries from summaries/{dataset}/gold_summaries.jsonl instead of original datasets")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable outlier filtering (keep all summaries including extremely long ones)")
    parser.add_argument("--raw", action="store_true",
                        help="Input file is in raw GPT output format (with custom_id and nested response structure)")
    
    args = parser.parse_args()
    
    try:
        file_path = Path(args.file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        
        dataset_name = extract_dataset_from_path(file_path)
        
        print(f"Processing {file_path}...", file=sys.stderr)
        print(f"Dataset: {dataset_name}", file=sys.stderr)
        print(f"Using {'local gold summaries' if args.gold_local else 'original dataset references'}", file=sys.stderr)
        
        result = evaluate_file(file_path, dataset_name, args.gold_local, filter_outliers=not args.no_filter, raw_format=args.raw)
        
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results also saved to {args.output}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()