#!/usr/bin/env python3
"""
Stage 1 Evaluation Script for Dermatology Domain Adaptation
==========================================================

This script evaluates the Stage 1 trained model on each dataset separately:
- DDI, Fitzpatrick17k, SCIN, DermNet
- Computes accuracy, precision, recall, F1-score for each dataset
- Generates confusion matrix for each dataset
- Provides detailed analysis of model performance per dataset
- Creates visualizations for each dataset evaluation

Key Features:
- Separate evaluation for each dataset
- Comprehensive evaluation metrics per dataset
- Cross-dataset performance comparison
- Confusion matrix analysis per dataset
- Performance visualization per dataset
- Detailed error analysis per dataset
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DermatologyEvaluator:
    """Evaluator for dermatology domain adaptation models"""
    
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.model, self.processor = self._load_model()
        
        # Evaluation results
        self.results = {}
        
    def _load_model(self):
        """Load the trained model and processor"""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Load base model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Load LoRA weights if they exist
        if (self.model_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, self.model_path)
            logger.info("✓ LoRA weights loaded")
        
        model.eval()
        return model, processor
    
    def load_dataset_data(self, data_dir: str, dataset_name: str) -> List[Dict]:
        """Load test data for a specific dataset"""
        # Load the unified data first
        test_file = os.path.join(data_dir, "test.jsonl")
        if not os.path.exists(test_file):
            # If no test file, use validation data
            test_file = os.path.join(data_dir, "val.jsonl")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data not found in {data_dir}")
        
        logger.info(f"Loading test data from: {test_file}")
        
        all_data = []
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
        
        # Filter data for specific dataset
        dataset_data = []
        for item in all_data:
            # Extract dataset name from image path
            image_path = item.get('image', '')
            if self._get_dataset_from_path(image_path) == dataset_name:
                dataset_data.append(item)
        
        logger.info(f"Loaded {len(dataset_data)} samples for {dataset_name} dataset")
        return dataset_data
    
    def _get_dataset_from_path(self, image_path: str) -> str:
        """Extract dataset name from image path"""
        path_lower = image_path.lower()
        
        if 'ddidiversedermatologyimages' in path_lower or 'ddi' in path_lower:
            return 'ddi'
        elif 'fitzpatrick17k' in path_lower or 'fitzpatrick' in path_lower:
            return 'fitzpatrick'
        elif 'scin' in path_lower:
            return 'scin'
        elif 'dermnet' in path_lower or 'kagglehub' in path_lower:
            return 'dermnet'
        else:
            return 'unknown'
    
    def predict_single(self, image_path: str, question: str = "What skin condition is shown in this image?") -> str:
        """Predict disease for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = self._upscale_if_needed(image)
            
            # Process input
            inputs = self.processor(
                text="<image>\n" + question,
                images=image,
                size={"shortest_edge": 672, "longest_edge": 672},
                do_resize=True,
                padding=False,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode output
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (everything after the question)
            answer = generated_text.split(question)[-1].strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error predicting for {image_path}: {e}")
            return "Error"
    
    def _upscale_if_needed(self, image: Image.Image, min_size: int = 672) -> Image.Image:
        """Upscale image if needed"""
        width, height = image.size
        if width < min_size or height < min_size:
            image = image.resize((min_size, min_size), Image.BICUBIC)
        return image
    
    def extract_disease_from_answer(self, answer: str) -> str:
        """Extract disease name from model answer"""
        # Simple extraction - look for common patterns
        answer_lower = answer.lower()
        
        # Common patterns
        patterns = [
            "this image shows",
            "this is",
            "the image shows",
            "shows",
            "appears to be",
            "looks like",
            "is",
        ]
        
        for pattern in patterns:
            if pattern in answer_lower:
                # Extract text after pattern
                start_idx = answer_lower.find(pattern) + len(pattern)
                disease = answer[start_idx:].strip()
                
                # Clean up the disease name
                disease = disease.replace(".", "").replace(",", "").strip()
                
                # Take only the first part (before any additional text)
                disease = disease.split()[0] if disease else ""
                
                return disease
        
        # If no pattern found, return the first word
        return answer.split()[0] if answer else "Unknown"
    
    def evaluate_dataset(self, test_data: List[Dict], dataset_name: str) -> Dict:
        """Evaluate model on a specific dataset"""
        logger.info(f"Evaluating on {dataset_name} dataset...")
        
        predictions = []
        true_labels = []
        errors = []
        
        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {dataset_name}")):
            try:
                # Get true label
                true_label = sample["conversations"][1]["value"]
                true_disease = self.extract_disease_from_answer(true_label)
                
                # Get prediction
                question = sample["conversations"][0]["value"]
                image_path = sample["image"]
                
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                predicted_answer = self.predict_single(image_path, question)
                predicted_disease = self.extract_disease_from_answer(predicted_answer)
                
                predictions.append(predicted_disease)
                true_labels.append(true_disease)
                
                # Log errors for analysis
                if predicted_disease.lower() != true_disease.lower():
                    errors.append({
                        "image": image_path,
                        "true_label": true_disease,
                        "predicted_label": predicted_disease,
                        "true_answer": true_label,
                        "predicted_answer": predicted_answer,
                    })
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Get unique labels for confusion matrix
        all_labels = sorted(list(set(true_labels + predictions)))
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=all_labels)
        
        # Create classification report
        report = classification_report(true_labels, predictions, labels=all_labels, zero_division=0)
        
        results = {
            "dataset_name": dataset_name,
            "total_samples": len(test_data),
            "evaluated_samples": len(predictions),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "labels": all_labels,
            "errors": errors,
            "classification_report": report,
        }
        
        logger.info(f"✓ {dataset_name} evaluation complete:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-score: {f1:.4f}")
        
        return results
    
    def create_visualizations(self, results: Dict, output_dir: str):
        """Create visualization plots for a specific dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        dataset_name = results["dataset_name"]
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = np.array(results["confusion_matrix"])
        labels = results["labels"]
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_title(f'Confusion Matrix - {dataset_name.upper()} Dataset')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_dir / f'confusion_matrix_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metrics Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [results["accuracy"], results["precision"], results["recall"], results["f1_score"]]
        
        bars = ax.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title(f'Evaluation Metrics - {dataset_name.upper()} Dataset')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'metrics_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error Analysis
        if results["errors"]:
            error_df = pd.DataFrame(results["errors"])
            
            # Most common errors
            error_counts = error_df.groupby(['true_label', 'predicted_label']).size().reset_index(name='count')
            error_counts = error_counts.sort_values('count', ascending=False).head(20)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(data=error_counts, x='count', y='true_label', hue='predicted_label', ax=ax)
            ax.set_title(f'Most Common Errors - {dataset_name.upper()} Dataset')
            ax.set_xlabel('Number of Errors')
            ax.set_ylabel('True Label')
            plt.tight_layout()
            plt.savefig(output_dir / f'error_analysis_{dataset_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"✓ Visualizations saved to: {output_dir}")
    
    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results for a specific dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        dataset_name = results["dataset_name"]
        
        # Save detailed results
        results_file = output_dir / f"evaluation_results_{dataset_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            "dataset_name": dataset_name,
            "total_samples": results["total_samples"],
            "evaluated_samples": results["evaluated_samples"],
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
            "num_errors": len(results["errors"]),
        }
        
        summary_file = output_dir / f"evaluation_summary_{dataset_name}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save classification report
        report_file = output_dir / f"classification_report_{dataset_name}.txt"
        with open(report_file, 'w') as f:
            f.write(results["classification_report"])
        
        logger.info(f"✓ Results saved to: {output_dir}")
    
    def create_cross_dataset_comparison(self, all_results: List[Dict], output_dir: str):
        """Create cross-dataset comparison visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create comparison DataFrame
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                'Dataset': result['dataset_name'].upper(),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'Samples': result['evaluated_samples']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Dataset Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i//2, i%2]
            bars = ax.bar(comparison_df['Dataset'], comparison_df[metric], color=color)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, comparison_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison table
        comparison_df.to_csv(output_dir / 'cross_dataset_comparison.csv', index=False)
        
        logger.info(f"✓ Cross-dataset comparison saved to: {output_dir}")
    
    def run_evaluation(self, data_dir: str, output_dir: str):
        """Run complete evaluation pipeline on all datasets"""
        logger.info("Starting Stage 1 Model Evaluation on All Datasets")
        
        # Define datasets to evaluate
        datasets = ['ddi', 'fitzpatrick', 'scin', 'dermnet']
        all_results = []
        
        for dataset_name in datasets:
            try:
                # Load dataset-specific data
                test_data = self.load_dataset_data(data_dir, dataset_name)
                
                if len(test_data) == 0:
                    logger.warning(f"No data found for {dataset_name} dataset, skipping...")
                    continue
                
                # Run evaluation
                results = self.evaluate_dataset(test_data, dataset_name)
                all_results.append(results)
                
                # Save results
                self.save_results(results, output_dir)
                
                # Create visualizations
                self.create_visualizations(results, output_dir)
                
            except Exception as e:
                logger.error(f"Error evaluating {dataset_name} dataset: {e}")
                continue
        
        # Create cross-dataset comparison
        if len(all_results) > 1:
            self.create_cross_dataset_comparison(all_results, output_dir)
        
        # Print final summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY - ALL DATASETS")
        print("="*80)
        
        for result in all_results:
            print(f"\n{result['dataset_name'].upper()} Dataset:")
            print(f"  Samples: {result['evaluated_samples']}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall: {result['recall']:.4f}")
            print(f"  F1-score: {result['f1_score']:.4f}")
            print(f"  Errors: {len(result['errors'])}")
        
        print("="*80)
        
        return all_results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Stage 1 Model Evaluation on All Datasets")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_dir", required=True, help="Path to test data directory")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model name")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DermatologyEvaluator(args.model_path, args.base_model)
    
    # Run evaluation on all datasets
    results = evaluator.run_evaluation(args.data_dir, args.output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
    print(f"Evaluated {len(results)} datasets")

if __name__ == "__main__":
    main()
