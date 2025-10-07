#!/usr/bin/env python3
"""
Stage 2 Evaluation Script for Dermatology Educational Alignment

This script evaluates the Stage 2 model's ability to provide comprehensive
educational responses including diagnosis, symptoms, precautions, education, and questions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class Stage2EducationalEvaluator:
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        
        # Keywords to identify different components in responses
        self.component_keywords = {
            "diagnosis": [
                "appears to be", "diagnosis", "consistent with", "suggests", 
                "likely", "characteristics", "presentation", "lesion shows"
            ],
            "symptoms": [
                "symptoms", "signs", "experience", "may include", "common",
                "typical", "itching", "pain", "burning", "discomfort"
            ],
            "precautions": [
                "precautions", "avoid", "important", "note", "safety",
                "considerations", "protect", "monitor", "seek medical"
            ],
            "education": [
                "is a", "condition", "disease", "caused by", "results from",
                "educational", "understand", "note", "chronic", "inflammatory"
            ],
            "questions": [
                "questions", "tell me", "clarify", "information", "helpful",
                "how long", "have you", "are you", "do you", "what"
            ],
            "disclaimer": [
                "medical advice", "consult", "healthcare", "professional",
                "diagnosis", "treatment", "educational purposes", "replace"
            ]
        }
        
        # Quality indicators
        self.quality_indicators = {
            "comprehensive": ["detailed", "comprehensive", "thorough", "complete"],
            "educational": ["educational", "understand", "explain", "information"],
            "safe": ["precautions", "safety", "medical advice", "consult"],
            "helpful": ["helpful", "assist", "guidance", "support"]
        }
    
    def load_test_data(self) -> List[Dict]:
        """Load test data for evaluation"""
        test_data = []
        
        # Load validation data
        val_file = self.test_data_path / "val.jsonl"
        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    test_data.append(json.loads(line.strip()))
        
        print(f"✓ Loaded {len(test_data)} test samples")
        return test_data
    
    def evaluate_response_components(self, response: str) -> Dict[str, bool]:
        """Evaluate if response contains different educational components"""
        response_lower = response.lower()
        components = {}
        
        for component, keywords in self.component_keywords.items():
            components[component] = any(keyword in response_lower for keyword in keywords)
        
        return components
    
    def evaluate_response_quality(self, response: str) -> Dict[str, float]:
        """Evaluate response quality indicators"""
        response_lower = response.lower()
        quality_scores = {}
        
        for quality, indicators in self.quality_indicators.items():
            score = sum(1 for indicator in indicators if indicator in response_lower)
            quality_scores[quality] = score / len(indicators)
        
        return quality_scores
    
    def evaluate_response_length(self, response: str) -> Dict[str, int]:
        """Evaluate response length metrics"""
        words = response.split()
        sentences = response.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "char_count": len(response)
        }
    
    def evaluate_medical_safety(self, response: str) -> Dict[str, bool]:
        """Evaluate medical safety aspects"""
        response_lower = response.lower()
        
        safety_checks = {
            "has_disclaimer": any(term in response_lower for term in [
                "medical advice", "consult", "healthcare", "professional"
            ]),
            "avoids_diagnosis": "diagnosis" in response_lower and "consult" in response_lower,
            "encourages_professional_care": any(term in response_lower for term in [
                "seek medical", "consult", "healthcare provider", "doctor"
            ]),
            "educational_purpose": "educational" in response_lower
        }
        
        return safety_checks
    
    def evaluate_disease_specific_content(self, response: str, disease: str) -> Dict[str, bool]:
        """Evaluate disease-specific content accuracy"""
        response_lower = response.lower()
        disease_lower = disease.lower()
        
        # Disease-specific checks
        disease_checks = {
            "mentions_disease": disease_lower in response_lower,
            "provides_context": len(response.split()) > 50,  # Substantial response
            "includes_management": any(term in response_lower for term in [
                "treatment", "management", "care", "precautions"
            ])
        }
        
        return disease_checks
    
    def run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive evaluation on test data"""
        print(f"\n" + "="*80)
        print("RUNNING COMPREHENSIVE STAGE 2 EVALUATION")
        print("="*80)
        
        results = {
            "component_analysis": defaultdict(list),
            "quality_analysis": defaultdict(list),
            "length_analysis": defaultdict(list),
            "safety_analysis": defaultdict(list),
            "disease_analysis": defaultdict(list),
            "overall_scores": {}
        }
        
        for i, sample in enumerate(test_data):
            if i % 100 == 0:
                print(f"Evaluating sample {i+1}/{len(test_data)}...")
            
            # Get response from conversations
            response = ""
            for conv in sample.get("conversations", []):
                if conv.get("from") == "gpt":
                    response = conv.get("value", "")
                    break
            
            if not response:
                continue
            
            # Evaluate components
            components = self.evaluate_response_components(response)
            for component, present in components.items():
                results["component_analysis"][component].append(present)
            
            # Evaluate quality
            quality = self.evaluate_response_quality(response)
            for quality_type, score in quality.items():
                results["quality_analysis"][quality_type].append(score)
            
            # Evaluate length
            length_metrics = self.evaluate_response_length(response)
            for metric, value in length_metrics.items():
                results["length_analysis"][metric].append(value)
            
            # Evaluate safety
            safety = self.evaluate_medical_safety(response)
            for safety_check, passed in safety.items():
                results["safety_analysis"][safety_check].append(passed)
            
            # Evaluate disease-specific content
            disease = sample.get("metadata", {}).get("disease", "unknown")
            disease_content = self.evaluate_disease_specific_content(response, disease)
            for check, passed in disease_content.items():
                results["disease_analysis"][check].append(passed)
        
        # Calculate overall scores
        results["overall_scores"] = self._calculate_overall_scores(results)
        
        return results
    
    def _calculate_overall_scores(self, results: Dict) -> Dict[str, float]:
        """Calculate overall evaluation scores"""
        scores = {}
        
        # Component coverage score
        component_scores = []
        for component, values in results["component_analysis"].items():
            if component != "disclaimer":  # Exclude disclaimer from coverage
                component_scores.append(np.mean(values))
        scores["component_coverage"] = np.mean(component_scores)
        
        # Quality score
        quality_scores = []
        for quality_type, values in results["quality_analysis"].items():
            quality_scores.append(np.mean(values))
        scores["quality_score"] = np.mean(quality_scores)
        
        # Safety score
        safety_scores = []
        for safety_check, values in results["safety_analysis"].items():
            safety_scores.append(np.mean(values))
        scores["safety_score"] = np.mean(safety_scores)
        
        # Disease-specific score
        disease_scores = []
        for check, values in results["disease_analysis"].items():
            disease_scores.append(np.mean(values))
        scores["disease_specific_score"] = np.mean(disease_scores)
        
        # Overall score
        scores["overall_score"] = np.mean([
            scores["component_coverage"],
            scores["quality_score"],
            scores["safety_score"],
            scores["disease_specific_score"]
        ])
        
        return scores
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        report = []
        report.append("# Stage 2 Educational Alignment Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Overall scores
        report.append("## Overall Performance Scores")
        report.append("")
        for score_name, score_value in results["overall_scores"].items():
            report.append(f"- **{score_name.replace('_', ' ').title()}**: {score_value:.3f}")
        report.append("")
        
        # Component analysis
        report.append("## Component Coverage Analysis")
        report.append("")
        for component, values in results["component_analysis"].items():
            coverage = np.mean(values) * 100
            report.append(f"- **{component.title()}**: {coverage:.1f}% coverage")
        report.append("")
        
        # Quality analysis
        report.append("## Response Quality Analysis")
        report.append("")
        for quality_type, values in results["quality_analysis"].items():
            avg_score = np.mean(values)
            report.append(f"- **{quality_type.title()}**: {avg_score:.3f} average score")
        report.append("")
        
        # Safety analysis
        report.append("## Medical Safety Analysis")
        report.append("")
        for safety_check, values in results["safety_analysis"].items():
            compliance = np.mean(values) * 100
            report.append(f"- **{safety_check.replace('_', ' ').title()}**: {compliance:.1f}% compliance")
        report.append("")
        
        # Length analysis
        report.append("## Response Length Analysis")
        report.append("")
        for metric, values in results["length_analysis"].items():
            avg_value = np.mean(values)
            report.append(f"- **{metric.replace('_', ' ').title()}**: {avg_value:.1f} average")
        report.append("")
        
        # Disease-specific analysis
        report.append("## Disease-Specific Content Analysis")
        report.append("")
        for check, values in results["disease_analysis"].items():
            compliance = np.mean(values) * 100
            report.append(f"- **{check.replace('_', ' ').title()}**: {compliance:.1f}% compliance")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        if results["overall_scores"]["overall_score"] < 0.7:
            report.append("- **Overall score is below 0.7**: Consider additional training or data augmentation")
        if results["overall_scores"]["safety_score"] < 0.8:
            report.append("- **Safety score is below 0.8**: Emphasize medical disclaimers and professional care recommendations")
        if results["overall_scores"]["component_coverage"] < 0.6:
            report.append("- **Component coverage is below 0.6**: Ensure all educational components are included in responses")
        if results["overall_scores"]["quality_score"] < 0.6:
            report.append("- **Quality score is below 0.6**: Improve response comprehensiveness and educational value")
        
        return "\n".join(report)
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: Path):
        """Save evaluation results to files"""
        # Save detailed results
        results_file = output_path / "stage2_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save report
        report = self.generate_evaluation_report(results)
        report_file = output_path / "stage2_evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Evaluation results saved to: {results_file}")
        print(f"✓ Evaluation report saved to: {report_file}")

def main():
    """Main evaluation function"""
    print("Stage 2 Educational Alignment Evaluation")
    print("=" * 50)
    
    # Configuration
    model_path = "stage2_output/checkpoint-1000"  # Update with actual model path
    test_data_path = Path("stage2_data")
    output_path = Path("stage2_evaluation")
    output_path.mkdir(exist_ok=True)
    
    # Create evaluator
    evaluator = Stage2EducationalEvaluator(model_path, test_data_path)
    
    # Load test data
    test_data = evaluator.load_test_data()
    
    if not test_data:
        print("❌ No test data found!")
        return
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(test_data)
    
    # Save results
    evaluator.save_evaluation_results(results, output_path)
    
    # Print summary
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Overall Score: {results['overall_scores']['overall_score']:.3f}")
    print(f"Component Coverage: {results['overall_scores']['component_coverage']:.3f}")
    print(f"Quality Score: {results['overall_scores']['quality_score']:.3f}")
    print(f"Safety Score: {results['overall_scores']['safety_score']:.3f}")
    print(f"Disease-Specific Score: {results['overall_scores']['disease_specific_score']:.3f}")
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
