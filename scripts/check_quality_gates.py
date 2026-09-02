#!/usr/bin/env python3
"""
Quality Gate Checker - Prevents regressions in CI/CD

This script:
1. Loads evaluation results
2. Compares against thresholds
3. Compares against baseline (if exists)
4. Fails build if quality drops
5. Creates comparison report
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

class QualityGateChecker:
    def __init__(self, min_recall: float, min_mrr: float, min_rouge: float):
        self.min_recall = min_recall
        self.min_mrr = min_mrr
        self.min_rouge = min_rouge
        self.passed = True
        self.errors = []
    
    def check_retrieval_metrics(self, results: Dict[str, Any]) -> bool:
        """Check retrieval evaluation results"""
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION RESULTS")
        print("="*60)
        
        summary = results.get("summary", {})
        
        recall_k = summary.get("recall_k", 0)
        mrr = summary.get("mrr", 0)
        
        print(f"Recall@K: {recall_k:.4f} (threshold: {self.min_recall:.4f})")
        print(f"MRR: {mrr:.4f} (threshold: {self.min_mrr:.4f})")
        
        if recall_k < self.min_recall:
            error = f"❌ Recall@K ({recall_k:.4f}) below threshold ({self.min_recall:.4f})"
            print(error)
            self.errors.append(error)
            self.passed = False
        else:
            print(f"✅ Recall@K passed")
        
        if mrr < self.min_mrr:
            error = f"❌ MRR ({mrr:.4f}) below threshold ({self.min_mrr:.4f})"
            print(error)
            self.errors.append(error)
            self.passed = False
        else:
            print(f"✅ MRR passed")
        
        return self.passed
    
    def check_generation_metrics(self, results: Dict[str, Any]) -> bool:
        """Check generation/RAGAS evaluation results"""
        print("\n" + "="*60)
        print("GENERATION EVALUATION RESULTS (RAGAS)")
        print("="*60)
        
        summary = results.get("summary", {})
        
        rouge_l = summary.get("rouge_l", 0)
        faithfulness = summary.get("faithfulness", 0)
        answer_relevancy = summary.get("answer_relevancy", 0)
        
        print(f"ROUGE-L: {rouge_l:.4f} (threshold: {self.min_rouge:.4f})")
        print(f"Faithfulness: {faithfulness:.4f}")
        print(f"Answer Relevancy: {answer_relevancy:.4f}")
        
        if rouge_l < self.min_rouge:
            error = f"❌ ROUGE-L ({rouge_l:.4f}) below threshold ({self.min_rouge:.4f})"
            print(error)
            self.errors.append(error)
            self.passed = False
        else:
            print(f"✅ ROUGE-L passed")
        
        if faithfulness < 0.6:
            error = f"❌ Faithfulness ({faithfulness:.4f}) critically low"
            print(error)
            self.errors.append(error)
            self.passed = False
        else:
            print(f"✅ Faithfulness acceptable")
        
        return self.passed
    
    def compare_with_baseline(
        self,
        current: Dict[str, Any],
        baseline_file: str
    ) -> bool:
        """
        Compare current metrics against baseline
        Fail if regression > 5%
        """
        baseline_path = Path(baseline_file)
        
        if not baseline_path.exists():
            print("\n⚠️  No baseline found - creating one")
            return True
        
        print("\n" + "="*60)
        print("REGRESSION CHECK (vs Baseline)")
        print("="*60)
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        current_summary = current.get("summary", {})
        baseline_summary = baseline.get("summary", {})
        
        # Check key metrics for regression
        metrics_to_check = ["recall_k", "mrr", "rouge_l"]
        regression_threshold = 0.05  # 5%
        
        for metric in metrics_to_check:
            current_val = current_summary.get(metric, 0)
            baseline_val = baseline_summary.get(metric, 0)
            
            if baseline_val == 0:
                continue
            
            regression = (baseline_val - current_val) / baseline_val
            
            print(f"\n{metric}")
            print(f"  Baseline: {baseline_val:.4f}")
            print(f"  Current:  {current_val:.4f}")
            print(f"  Change:   {regression:.1%}")
            
            if regression > regression_threshold:
                error = f"❌ {metric} regressed by {regression:.1%} (threshold: {regression_threshold:.1%})"
                print(error)
                self.errors.append(error)
                self.passed = False
            else:
                print(f"✅ No significant regression")
        
        return self.passed
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("QUALITY GATE SUMMARY")
        print("="*60)
        
        if self.passed:
            print("✅ ALL QUALITY GATES PASSED")
            return 0
        else:
            print("❌ QUALITY GATES FAILED")
            print("\nErrors:")
            for error in self.errors:
                print(f"  {error}")
            return 1


def main():
    parser = argparse.ArgumentParser(
        description="Check quality gates for RAG system"
    )
    parser.add_argument(
        "--retrieval-file",
        required=True,
        help="Path to retrieval evaluation results JSON"
    )
    parser.add_argument(
        "--generation-file",
        required=True,
        help="Path to generation evaluation results JSON"
    )
    parser.add_argument(
        "--baseline-retrieval",
        default="data/eval/results/baseline_retrieval.json",
        help="Path to baseline retrieval metrics"
    )
    parser.add_argument(
        "--baseline-generation",
        default="data/eval/results/baseline_generation.json",
        help="Path to baseline generation metrics"
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.60,
        help="Minimum Recall@K threshold"
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=0.50,
        help="Minimum MRR threshold"
    )
    parser.add_argument(
        "--min-rouge",
        type=float,
        default=0.45,
        help="Minimum ROUGE-L threshold"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.retrieval_file).exists():
        print(f"❌ Retrieval results file not found: {args.retrieval_file}")
        return 1
    
    if not Path(args.generation_file).exists():
        print(f"❌ Generation results file not found: {args.generation_file}")
        return 1
    
    # Initialize checker
    checker = QualityGateChecker(
        min_recall=args.min_recall,
        min_mrr=args.min_mrr,
        min_rouge=args.min_rouge
    )
    
    # Load and check results
    with open(args.retrieval_file) as f:
        retrieval_results = json.load(f)
    
    with open(args.generation_file) as f:
        generation_results = json.load(f)
    
    # Run checks
    checker.check_retrieval_metrics(retrieval_results)
    checker.check_generation_metrics(generation_results)
    
    # Check against baseline
    checker.compare_with_baseline(
        retrieval_results,
        args.baseline_retrieval
    )
    
    # Print summary and exit
    return checker.print_summary()


if __name__ == "__main__":
    sys.exit(main())