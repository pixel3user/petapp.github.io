#!/usr/bin/env python3
"""
Stage 2 Data Preparation Pipeline for Dermatology Educational Alignment

This script creates a small but rich dataset for Stage 2 training where each sample contains:
- Diagnosis
- Symptoms
- Precautions/advice
- Educational explanation
- Clarifying questions

The goal is to teach the model to provide comprehensive, educational responses
about dermatological conditions with proper medical guidance.
"""

import pandas as pd
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict

class Stage2EducationalDataProcessor:
    def __init__(self, data_root: str = "data", output_dir: str = "stage2_data"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Educational templates for different types of responses
        self.educational_templates = {
            "diagnosis": [
                "Based on the visual characteristics and your description, this appears to be {disease}.",
                "The clinical presentation suggests {disease} as the most likely diagnosis.",
                "This lesion shows features consistent with {disease}."
            ],
            "symptoms": [
                "Common symptoms of {disease} include: {symptoms}",
                "You may experience: {symptoms}",
                "Typical signs and symptoms are: {symptoms}"
            ],
            "precautions": [
                "Important precautions: {precautions}",
                "Please note: {precautions}",
                "Safety considerations: {precautions}"
            ],
            "education": [
                "{disease} is {education}",
                "To help you understand: {disease} is {education}",
                "Educational note: {disease} is {education}"
            ],
            "questions": [
                "To better assist you, could you tell me: {questions}",
                "Additional information that would be helpful: {questions}",
                "Please clarify: {questions}"
            ]
        }
        
        # Disease-specific educational content
        self.disease_education = {
            "psoriasis": {
                "symptoms": "red, scaly patches, itching, burning sensation, dry skin, joint pain",
                "precautions": "avoid triggers like stress, smoking, alcohol; use gentle skincare; avoid scratching",
                "education": "a chronic autoimmune condition that causes rapid skin cell turnover, leading to thick, scaly patches",
                "questions": "How long have you had these patches? Do you have a family history of psoriasis? Are you experiencing any joint pain?"
            },
            "eczema": {
                "symptoms": "dry, itchy skin, red or brown patches, small raised bumps, thickened skin",
                "precautions": "avoid harsh soaps, use fragrance-free products, keep skin moisturized, avoid known allergens",
                "education": "a chronic inflammatory skin condition that causes dry, itchy, and inflamed skin",
                "questions": "What triggers seem to worsen your symptoms? Have you tried any treatments before? Do you have allergies?"
            },
            "melanoma": {
                "symptoms": "new or changing mole, irregular borders, multiple colors, large size, asymmetry",
                "precautions": "seek immediate medical evaluation, avoid sun exposure, use sunscreen, monitor for changes",
                "education": "a serious form of skin cancer that can spread to other parts of the body if not treated early",
                "questions": "How long has this lesion been present? Has it changed in size, color, or shape? Do you have a family history of skin cancer?"
            },
            "acne": {
                "symptoms": "pimples, blackheads, whiteheads, cysts, oily skin, inflammation",
                "precautions": "avoid picking or squeezing, use non-comedogenic products, gentle cleansing, avoid excessive scrubbing",
                "education": "a common skin condition caused by clogged hair follicles, excess oil production, and bacteria",
                "questions": "What skincare products are you currently using? How long have you had acne? Have you tried any treatments?"
            },
            "basal cell carcinoma": {
                "symptoms": "pearly or waxy bump, flat flesh-colored lesion, bleeding or scabbing sore",
                "precautions": "seek medical evaluation, protect from sun exposure, regular skin checks",
                "education": "the most common type of skin cancer, usually slow-growing and rarely spreads",
                "questions": "How long has this lesion been present? Has it changed recently? Do you have a history of sun exposure?"
            },
            "squamous cell carcinoma": {
                "symptoms": "firm red nodule, flat lesion with scaly crust, persistent sore that won't heal",
                "precautions": "seek immediate medical evaluation, avoid sun exposure, regular skin monitoring",
                "education": "a type of skin cancer that can grow quickly and may spread if not treated",
                "questions": "How long has this been present? Has it grown or changed? Do you have a history of sun damage?"
            },
            "seborrheic keratosis": {
                "symptoms": "waxy, stuck-on appearance, brown or black color, rough texture, various sizes",
                "precautions": "usually benign, but monitor for changes, avoid picking or scratching",
                "education": "a common benign skin growth that appears as waxy, stuck-on lesions",
                "questions": "How long have you had these growths? Have they changed in appearance? Do they cause any discomfort?"
            },
            "tinea": {
                "symptoms": "red, scaly patches, itching, ring-shaped appearance, possible blisters",
                "precautions": "keep area clean and dry, avoid sharing personal items, use antifungal treatment",
                "education": "a fungal infection of the skin, also known as ringworm",
                "questions": "How long have you had this rash? Have you been in contact with anyone who has similar symptoms? Are you using any treatments?"
            },
            "urticaria": {
                "symptoms": "raised, itchy welts, red or skin-colored, various sizes, may come and go",
                "precautions": "identify and avoid triggers, use antihistamines, seek medical care if severe",
                "education": "an allergic reaction that causes raised, itchy welts on the skin",
                "questions": "What do you think triggered this reaction? Have you eaten anything new? Are you taking any new medications?"
            },
            "contact dermatitis": {
                "symptoms": "red, itchy rash, blisters, dry, cracked skin, burning sensation",
                "precautions": "identify and avoid the irritant, use gentle skincare, apply cool compresses",
                "education": "an inflammatory skin reaction caused by contact with an irritant or allergen",
                "questions": "What products have you used recently? Have you been exposed to any new substances? When did the rash first appear?"
            }
        }
        
        # Load Stage 1 data to get disease distribution
        self.stage1_data = self._load_stage1_data()
        
    def _load_stage1_data(self) -> pd.DataFrame:
        """Load Stage 1 unified dataset to understand disease distribution"""
        stage1_file = Path("stage1_data/unified_dataset.csv")
        if stage1_file.exists():
            df = pd.read_csv(stage1_file)
            print(f"✓ Loaded Stage 1 data: {len(df)} samples, {df['disease'].nunique()} diseases")
            return df
        else:
            print("⚠ Stage 1 data not found, using default disease list")
            return pd.DataFrame()
    
    def create_educational_dataset(self, target_samples: int = 2000) -> List[Dict]:
        """Create a rich educational dataset with comprehensive information for each sample"""
        print(f"\n" + "="*80)
        print("CREATING STAGE 2 EDUCATIONAL DATASET")
        print("="*80)
        
        educational_samples = []
        
        # Get disease distribution from Stage 1 data
        if not self.stage1_data.empty:
            disease_counts = self.stage1_data['disease'].value_counts()
            # Focus on most common diseases for better coverage
            top_diseases = disease_counts.head(50).index.tolist()
        else:
            # Fallback to our educational content diseases
            top_diseases = list(self.disease_education.keys())
        
        print(f"Creating educational samples for {len(top_diseases)} diseases...")
        
        # Create samples for each disease
        for disease in top_diseases:
            # Get disease info from Stage 1 data
            disease_samples = self.stage1_data[self.stage1_data['disease'] == disease] if not self.stage1_data.empty else []
            
            # Determine number of samples for this disease (proportional to frequency)
            if not disease_samples.empty:
                num_samples = min(50, max(5, len(disease_samples) // 20))  # 5-50 samples per disease
            else:
                num_samples = 10  # Default for diseases not in Stage 1
            
            # Create educational samples for this disease
            for i in range(num_samples):
                sample = self._create_educational_sample(disease, disease_samples, i)
                if sample:
                    educational_samples.append(sample)
        
        # If we need more samples, create additional ones
        while len(educational_samples) < target_samples:
            disease = random.choice(top_diseases)
            disease_samples = self.stage1_data[self.stage1_data['disease'] == disease] if not self.stage1_data.empty else []
            sample = self._create_educational_sample(disease, disease_samples, len(educational_samples))
            if sample:
                educational_samples.append(sample)
        
        print(f"✓ Created {len(educational_samples)} educational samples")
        return educational_samples[:target_samples]
    
    def _create_educational_sample(self, disease: str, disease_samples: pd.DataFrame, sample_idx: int) -> Dict:
        """Create a single educational sample with rich information"""
        
        # Get image path from Stage 1 data if available
        if not disease_samples.empty:
            image_path = disease_samples.iloc[sample_idx % len(disease_samples)]['image_path']
        else:
            # Fallback image path (will need to be handled in training)
            image_path = f"placeholder_{disease}_{sample_idx}.jpg"
        
        # Get educational content for this disease
        education_info = self.disease_education.get(disease.lower(), self._get_default_education(disease))
        
        # Create comprehensive conversation
        conversation = self._create_educational_conversation(disease, education_info)
        
        return {
            "image": image_path,
            "conversations": conversation,
            "metadata": {
                "dataset": "stage2_educational",
                "disease": disease,
                "sample_type": "educational",
                "contains_diagnosis": True,
                "contains_symptoms": True,
                "contains_precautions": True,
                "contains_education": True,
                "contains_questions": True
            }
        }
    
    def _create_educational_conversation(self, disease: str, education_info: Dict) -> List[Dict]:
        """Create a comprehensive educational conversation"""
        
        # User input variations
        user_inputs = [
            "I have this skin condition. Can you help me understand what it is and what I should do?",
            "What can you tell me about this skin lesion? I'm concerned about it.",
            "I noticed this on my skin. Could you provide some information about it?",
            "Can you help me understand this skin condition and give me some guidance?",
            "What is this skin issue and what precautions should I take?"
        ]
        
        user_input = random.choice(user_inputs)
        
        # Create comprehensive response
        response_parts = []
        
        # 1. Diagnosis
        diagnosis_template = random.choice(self.educational_templates["diagnosis"])
        response_parts.append(diagnosis_template.format(disease=disease))
        
        # 2. Symptoms
        if education_info.get("symptoms"):
            symptoms_template = random.choice(self.educational_templates["symptoms"])
            response_parts.append(symptoms_template.format(
                disease=disease, 
                symptoms=education_info["symptoms"]
            ))
        
        # 3. Educational explanation
        if education_info.get("education"):
            education_template = random.choice(self.educational_templates["education"])
            response_parts.append(education_template.format(
                disease=disease,
                education=education_info["education"]
            ))
        
        # 4. Precautions/advice
        if education_info.get("precautions"):
            precautions_template = random.choice(self.educational_templates["precautions"])
            response_parts.append(precautions_template.format(
                precautions=education_info["precautions"]
            ))
        
        # 5. Clarifying questions
        if education_info.get("questions"):
            questions_template = random.choice(self.educational_templates["questions"])
            response_parts.append(questions_template.format(
                questions=education_info["questions"]
            ))
        
        # 6. Medical disclaimer
        response_parts.append("Please note: This information is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.")
        
        # Combine response parts
        model_response = " ".join(response_parts)
        
        return [
            {"from": "human", "value": f"<image>\n{user_input}"},
            {"from": "gpt", "value": model_response}
        ]
    
    def _get_default_education(self, disease: str) -> Dict:
        """Get default educational content for diseases not in our knowledge base"""
        return {
            "symptoms": "various skin changes, possible itching or discomfort",
            "precautions": "monitor for changes, avoid irritation, seek medical advice if concerned",
            "education": "a dermatological condition that requires proper medical evaluation",
            "questions": "How long have you had this condition? Have you noticed any changes? Are you experiencing any symptoms?"
        }
    
    def create_training_data(self, educational_samples: List[Dict], train_ratio: float = 0.8):
        """Create training and validation splits"""
        print(f"\n" + "="*80)
        print("CREATING TRAINING DATA SPLITS")
        print("="*80)
        
        # Shuffle samples
        random.shuffle(educational_samples)
        
        # Split into train/val
        split_idx = int(len(educational_samples) * train_ratio)
        train_data = educational_samples[:split_idx]
        val_data = educational_samples[split_idx:]
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        # Save training data
        train_file = self.output_dir / "train.jsonl"
        val_file = self.output_dir / "val.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Create metadata
        metadata = {
            "total_train_samples": len(train_data),
            "total_val_samples": len(val_data),
            "total_diseases": len(set(sample['metadata']['disease'] for sample in educational_samples)),
            "dataset_type": "stage2_educational",
            "features": [
                "diagnosis",
                "symptoms", 
                "precautions",
                "educational_explanation",
                "clarifying_questions"
            ],
            "disease_distribution": self._get_disease_distribution(educational_samples)
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Training data saved to: {train_file}")
        print(f"✓ Validation data saved to: {val_file}")
        print(f"✓ Metadata saved to: {metadata_file}")
        
        return train_data, val_data
    
    def _get_disease_distribution(self, samples: List[Dict]) -> Dict:
        """Get disease distribution in the dataset"""
        disease_counts = defaultdict(int)
        for sample in samples:
            disease = sample['metadata']['disease']
            disease_counts[disease] += 1
        
        return dict(sorted(disease_counts.items(), key=lambda x: x[1], reverse=True))
    
    def validate_data(self, train_file: Path, val_file: Path):
        """Validate the generated training data"""
        print(f"\n" + "="*80)
        print("VALIDATING TRAINING DATA")
        print("="*80)
        
        train_samples = 0
        val_samples = 0
        missing_images = 0
        invalid_conversations = 0
        
        # Validate train data
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    train_samples += 1
                    
                    # Check required fields
                    if 'image' not in data or 'conversations' not in data:
                        invalid_conversations += 1
                        continue
                    
                    # Check conversation format
                    if not isinstance(data['conversations'], list) or len(data['conversations']) != 2:
                        invalid_conversations += 1
                        continue
                    
                    # Check if image exists (skip URL validation for now)
                    if isinstance(data['image'], str) and not data['image'].startswith('http') and not os.path.exists(data['image']):
                        missing_images += 1
                        
                except json.JSONDecodeError:
                    invalid_conversations += 1
        
        # Validate val data
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    val_samples += 1
                    
                    # Check required fields
                    if 'image' not in data or 'conversations' not in data:
                        invalid_conversations += 1
                        continue
                    
                    # Check conversation format
                    if not isinstance(data['conversations'], list) or len(data['conversations']) != 2:
                        invalid_conversations += 1
                        continue
                    
                    # Check if image exists (skip URL validation for now)
                    if isinstance(data['image'], str) and not data['image'].startswith('http') and not os.path.exists(data['image']):
                        missing_images += 1
                        
                except json.JSONDecodeError:
                    invalid_conversations += 1
        
        print(f"Train samples: {train_samples}")
        print(f"Val samples: {val_samples}")
        print(f"Missing images: {missing_images}")
        print(f"Invalid conversations: {invalid_conversations}")
        
        if missing_images == 0 and invalid_conversations == 0:
            print("✅ All data validation checks passed!")
        else:
            print("⚠️ Some validation issues found, but data is usable for training")

def main():
    """Main function to run Stage 2 data preparation"""
    print("Stage 2 Data Preparation Pipeline for Dermatology Educational Alignment")
    print("="*80)
    
    # Create processor
    processor = Stage2EducationalDataProcessor()
    
    # Create educational dataset
    educational_samples = processor.create_educational_dataset(target_samples=2000)
    
    # Create training data
    train_data, val_data = processor.create_training_data(educational_samples)
    
    # Validate data
    train_file = processor.output_dir / "train.jsonl"
    val_file = processor.output_dir / "val.jsonl"
    processor.validate_data(train_file, val_file)
    
    print(f"\n" + "="*80)
    print("STAGE 2 DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {processor.output_dir}")
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")
    print(f"Metadata: {processor.output_dir / 'metadata.json'}")
    
    print(f"\nNext steps:")
    print(f"1. Review the generated educational conversations")
    print(f"2. Run Stage 2 training with the prepared data")
    print(f"3. Monitor training progress and adjust parameters as needed")
    
    print(f"\nEducational Features:")
    print(f"- Comprehensive diagnosis and explanation")
    print(f"- Symptom descriptions and management")
    print(f"- Safety precautions and advice")
    print(f"- Educational content about conditions")
    print(f"- Clarifying questions for better assessment")

if __name__ == "__main__":
    main()
