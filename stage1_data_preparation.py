#!/usr/bin/env python3
"""
Stage 1 Data Preparation Pipeline for Dermatology Domain Adaptation
================================================================

This script prepares the five datasets (DDI, Fitzpatrick17k, SCIN, DermNet, DermAVQA) for Stage 1 training:
- Image → Disease Label (short captions)
- Freeze vision encoder + LLM, train projection layer
- Format: LLaVA-style training data with synthetic conversations

NEW: Synthetic Conversation Generation + Real Conversations
- Creates realistic patient-doctor interactions
- Integrates available metadata (skin tone, demographics, symptoms)
- Generates multi-turn conversations with follow-up questions
- Includes clinical reasoning and safety recommendations
- Adds real conversations from DermAVQA with medical validation

Dataset Summary:
- DDI: 656 samples, 78 diseases (many with <20 samples - problematic)
- Fitzpatrick17k: 16,577 samples, 114 diseases (all have >50 samples - excellent)
- SCIN: 3,061 samples, 210 diseases (many with <20 samples - problematic)
- DermNet: 19,559 samples, 23 disease categories (all have >200 samples - excellent)
- DermAVQA: ~156 samples, real conversations (Chinese, English, Spanish - excellent quality)

Total: ~44,000+ samples across 612+ unique conditions
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import ast
from collections import Counter
import random
from PIL import Image
import shutil
import re

class DermatologyDataProcessor:
    def __init__(self, data_root: str = "data", output_dir: str = "stage1_data"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Minimum samples threshold for training
        self.min_samples_threshold = 20
        
        # Dataset paths
        self.dataset_paths = {
            'ddi': self.data_root / 'ddidiversedermatologyimages',
            'fitzpatrick': self.data_root / 'fitzpatrick17k',
            'scin': self.data_root / 'scin',
            'dermnet': Path('.cache/kagglehub/datasets/shubhamgoel27/dermnet/versions/1'),
            'dermavqa': self.data_root / 'dermavqa'
        }
        
        # Load datasets
        self.datasets = {}
        self.load_datasets()
        
        # Conversation templates for synthetic generation
        self.conversation_templates = [
            {
                "template": "I have this skin condition. I'm a {age_group} {sex}, and it's been there for about {duration}. It's on my {body_part} and it's {symptoms}. What could this be?",
                "symptoms": ["itchy", "painful", "burning", "no symptoms"]
            },
            {
                "template": "I noticed this on my {body_part} {duration} ago. I'm a {age_group} {sex}. It's {symptoms}. Should I be concerned?",
                "symptoms": ["getting bigger", "changing color", "bleeding", "not changing"]
            },
            {
                "template": "Can you help me identify this skin condition? I'm a {age_group} {sex}, and this appeared on my {body_part} {duration} ago. It's {symptoms}.",
                "symptoms": ["very itchy", "painful", "burning", "not bothering me"]
            },
            {
                "template": "I'm worried about this skin condition. I'm a {age_group} {sex}, and it's been on my {body_part} for {duration}. It's {symptoms}. What do you think?",
                "symptoms": ["getting worse", "not healing", "spreading", "causing discomfort"]
            }
        ]
        
        # Follow-up question templates
        self.follow_up_templates = [
            "Can you tell me more about this condition?",
            "What should I do next?",
            "Is this serious?",
            "Should I see a doctor?",
            "What causes this condition?",
            "Will it spread?",
            "How long will it last?",
            "What treatments are available?"
        ]
        
        # Tokens/strings to treat as unknown/useless
        self._unknown_tokens = {"unknown", "unknown condition", "unknown location", "none", "n/a", "na", ""}

    def _is_useless_value(self, value) -> bool:
        """Return True if value is None or one of predefined unknown/useless tokens (case-insensitive)."""
        if value is None:
            return True
        try:
            if isinstance(value, float) and np.isnan(value):
                return True
        except Exception:
            pass
        if isinstance(value, str):
            lowered = value.strip().lower()
            return lowered in self._unknown_tokens
        return False

    def _sanitize_text(self, text: str) -> str:
        """Remove common 'unknown' phrases and tidy whitespace/punctuation."""
        if not isinstance(text, str) or not text:
            return text
        cleaned = text
        # Remove specific bad phrases first
        cleaned = re.sub(r"\bunknown location\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bunknown ago\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"I'm a\s+unknown\s+unknown\.?", "", cleaned, flags=re.IGNORECASE)
        # Remove standalone 'unknown' tokens
        cleaned = re.sub(r"\bunknown\b", "", cleaned, flags=re.IGNORECASE)
        # Collapse multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Fix spaces before punctuation
        cleaned = re.sub(r"\s+([\.,;:!?])", r"\1", cleaned)
        # Remove stray spaces around newlines
        cleaned = re.sub(r" *\n *", "\n", cleaned)
        # Trim spaces at ends of lines
        cleaned = "\n".join([line.strip() for line in cleaned.split("\n")])
        return cleaned.strip()

    def _sanitize_item(self, item):
        """Clean unknown/useless values from a single LLaVA-style item. Return None to drop the item."""
        try:
            metadata = item.get('metadata', {})
            disease_value = metadata.get('disease', item.get('metadata', {}).get('label'))
            if self._is_useless_value(disease_value):
                return None

            # Clean metadata: drop keys with useless values
            for key in list(metadata.keys()):
                if self._is_useless_value(metadata.get(key)):
                    metadata.pop(key, None)
            item['metadata'] = metadata

            # Sanitize human and gpt texts
            if 'conversations' in item and isinstance(item['conversations'], list):
                # Human
                if len(item['conversations']) >= 1 and 'value' in item['conversations'][0]:
                    item['conversations'][0]['value'] = self._sanitize_text(item['conversations'][0]['value'])

                # GPT: remove lines in Clinical Context that contain unknown/empty after colon
                if len(item['conversations']) >= 2 and 'value' in item['conversations'][1]:
                    gpt_text = item['conversations'][1]['value']
                    # Process Clinical Context block if present
                    parts = gpt_text.split('\n')
                    output_lines = []
                    in_cc = False
                    cc_buffer = []
                    for line in parts:
                        if line.strip().startswith("**Clinical Context:**"):
                            in_cc = True
                            cc_buffer = [line]
                            continue
                        if in_cc:
                            if line.strip().startswith("**") and line.strip().endswith("**"):
                                # next section begins; flush filtered CC and then this heading
                                # Filter CC lines: keep only non-empty values after colon
                                kept = []
                                for cc_line in cc_buffer[1:]:
                                    if ":" in cc_line:
                                        rhs = cc_line.split(":", 1)[1].strip()
                                        if not self._is_useless_value(rhs) and rhs != "-":
                                            kept.append(cc_line)
                                if len(kept) > 0:
                                    output_lines.append(cc_buffer[0])
                                    output_lines.extend(kept)
                                    output_lines.append("")
                                in_cc = False
                                output_lines.append(line)
                            else:
                                cc_buffer.append(line)
                            continue
                        output_lines.append(line)
                    if in_cc:
                        kept = []
                        for cc_line in cc_buffer[1:]:
                            if ":" in cc_line:
                                rhs = cc_line.split(":", 1)[1].strip()
                                if not self._is_useless_value(rhs) and rhs != "-":
                                    kept.append(cc_line)
                        if len(kept) > 0:
                            output_lines.append(cc_buffer[0])
                            output_lines.extend(kept)
                    new_gpt = "\n".join(output_lines)
                    item['conversations'][1]['value'] = self._sanitize_text(new_gpt)

            return item
        except Exception:
            return item

    def load_datasets(self):
        """Load all four datasets"""
        print("Loading datasets...")
        
        # Load DDI
        ddi_metadata = self.dataset_paths['ddi'] / 'ddi_metadata.csv'
        if ddi_metadata.exists():
            self.datasets['ddi'] = pd.read_csv(ddi_metadata)
            print(f"✓ DDI: {len(self.datasets['ddi'])} samples")
        
        # Load Fitzpatrick17k
        fitz_metadata = self.dataset_paths['fitzpatrick'] / 'fitzpatrick17k.csv'
        if fitz_metadata.exists():
            self.datasets['fitzpatrick'] = pd.read_csv(fitz_metadata)
            print(f"✓ Fitzpatrick17k: {len(self.datasets['fitzpatrick'])} samples")
        
        # Load SCIN
        scin_cases = self.dataset_paths['scin'] / 'dataset' / 'scin_cases.csv'
        scin_labels = self.dataset_paths['scin'] / 'dataset' / 'scin_labels.csv'
        if scin_cases.exists() and scin_labels.exists():
            # Load cases (has image paths) and labels (has disease labels)
            cases_df = pd.read_csv(scin_cases)
            labels_df = pd.read_csv(scin_labels)
            
            # Merge cases and labels on case_id
            scin_df = cases_df.merge(labels_df, on='case_id', how='inner')
            
            # Extract most confident labels
            scin_df['most_confident_label'] = scin_df.apply(self._extract_most_confident_label, axis=1)
            self.datasets['scin'] = scin_df.dropna(subset=['most_confident_label'])
            print(f"✓ SCIN: {len(self.datasets['scin'])} samples with valid labels")
        
        # Load DermNet (analyze folder structure)
        dermnet_path = self.dataset_paths['dermnet']
        if dermnet_path.exists():
            self.datasets['dermnet'] = self._analyze_dermnet_structure(dermnet_path)
            print(f"✓ DermNet: {len(self.datasets['dermnet'])} samples")
        
        # Load SkinCap from local CSV (has labels and metadata)
        skincap_metadata = Path('data/skincap/skincap_v240623.csv')
        if skincap_metadata.exists():
            skincap_df = pd.read_csv(skincap_metadata)
            # Filter out images marked as "Do not consider this image" if column exists
            if 'Do not consider this image' in skincap_df.columns:
                skincap_df = skincap_df[skincap_df['Do not consider this image'] != 1]
            self.datasets['skincap'] = skincap_df
            print(f"✓ SkinCap: {len(self.datasets['skincap'])} samples")
        else:
            print("⚠ SkinCap: Local CSV file not found")
        
        # Load DermAVQA
        dermavqa_path = self.dataset_paths['dermavqa']
        if dermavqa_path.exists():
            self.datasets['dermavqa'] = self._load_dermavqa_data(dermavqa_path)
            print(f"✓ DermAVQA: {len(self.datasets['dermavqa'])} samples")
    
    def _extract_most_confident_label(self, row):
        """Extract most confident label from SCIN dataset"""
        try:
            labels = ast.literal_eval(row['dermatologist_skin_condition_on_label_name'])
            confidences = ast.literal_eval(row['dermatologist_skin_condition_confidence'])
            
            if not labels or not confidences:
                return None
                
            max_conf_idx = np.argmax(confidences)
            return labels[max_conf_idx]
            
        except (ValueError, SyntaxError, TypeError):
            return None
    
    def _analyze_dermnet_structure(self, dermnet_path: Path) -> pd.DataFrame:
        """Analyze DermNet folder structure and create DataFrame"""
        train_path = dermnet_path / "train"
        test_path = dermnet_path / "test"
        
        samples = []
        
        # Process train folder
        for disease_folder in train_path.iterdir():
            if disease_folder.is_dir():
                disease_name = disease_folder.name
                for image_file in disease_folder.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        samples.append({
                            'image_path': str(image_file),
                            'disease': disease_name,
                            'split': 'train'
                        })
        
        # Process test folder
        for disease_folder in test_path.iterdir():
            if disease_folder.is_dir():
                disease_name = disease_folder.name
                for image_file in disease_folder.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        samples.append({
                            'image_path': str(image_file),
                            'disease': disease_name,
                            'split': 'test'
                        })
        
        return pd.DataFrame(samples)
    
    def _load_dermavqa_data(self, dermavqa_path: Path) -> pd.DataFrame:
        """Load DermAVQA dataset with conversations and metadata"""
        samples = []
        
        # Load metadata
        metadata_file = dermavqa_path / 'data' / 'iiyi' / 'df_mediqa-m3g-final.csv'
        userinfo_file = dermavqa_path / 'data' / 'iiyi' / 'df_userinfo.csv'
        
        if not metadata_file.exists() or not userinfo_file.exists():
            print("⚠️ DermAVQA metadata files not found")
            return pd.DataFrame()
        
        df_metadata = pd.read_csv(metadata_file)
        df_userinfo = pd.read_csv(userinfo_file)
        
        # Load conversation data (valid and test)
        conversation_files = [
            dermavqa_path / 'data' / 'iiyi' / 'valid_ht_cleaned.json',
            dermavqa_path / 'data' / 'iiyi' / 'test_ht.json'
        ]
        
        for conv_file in conversation_files:
            if conv_file.exists():
                try:
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        conversations = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parsing error in {conv_file}: {e}")
                    print("Attempting to load with error recovery...")
                    # Try to load line by line and skip malformed entries
                    conversations = []
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Try to fix common JSON issues
                        content = content.replace('update in the eTravel system (https://etravel.gov.ph)', '')
                        content = content.replace('The following conditions and', '')
                        try:
                            conversations = json.loads(content)
                        except:
                            print(f"Could not recover JSON, skipping {conv_file}")
                            conversations = []
                
                for conv in conversations:
                    encounter_id = conv['encounter_id']
                    
                    # For now, use default metadata since encounter IDs don't match
                    # TODO: Fix encounter ID mapping between JSON and CSV files
                    encounter_metadata = {
                        'age': None,
                        'sex': None,
                        'anatomic_locations': '',
                        'author_role': 'patient'
                    }
                    
                    # Process each response
                    for response in conv.get('responses', []):
                        # Get user info for quality filtering
                        user_info = df_userinfo[df_userinfo['author_id'] == response['author_id']]
                        
                        # Filter for high-quality responses
                        if (response.get('completeness', 0) >= 0.5 and 
                            len(user_info) > 0 and 
                            user_info.iloc[0]['validation_level'] in ['md_validated', 'md1_validated', 'md2_validated', 'md3_validated', 'md4_validated', 'realid_validated']):
                            
                            # Get image path
                            image_path = self._get_dermavqa_image_path(encounter_id, conv['image_ids'][0], dermavqa_path)
                            
                            if image_path and os.path.exists(image_path):
                                sample = {
                                    'encounter_id': encounter_id,
                                    'image_path': image_path,
                                    'query_title_zh': conv.get('query_title_zh', ''),
                                    'query_content_zh': conv.get('query_content_zh', ''),
                                    'query_title_en': conv.get('query_title_en', ''),
                                    'query_content_en': conv.get('query_content_en', ''),
                                    'query_title_es': conv.get('query_title_es', ''),
                                    'query_content_es': conv.get('query_content_es', ''),
                                    'response_zh': response.get('content_zh', ''),
                                    'response_en': response.get('content_en', ''),
                                    'response_es': response.get('content_es', ''),
                                    'completeness': response.get('completeness', 0),
                                    'contains_freq_ans': response.get('contains_freq_ans', 0),
                                    'author_id': response['author_id'],
                                    'validation_level': user_info.iloc[0]['validation_level'],
                                    'rank_level': user_info.iloc[0]['rank_level'],
                                    'age': encounter_metadata.get('age_norm', None),
                                    'sex': encounter_metadata.get('sex', None),
                                    'anatomical_location': encounter_metadata.get('anatomic_locations', ''),
                                    'author_role': encounter_metadata.get('author_role', ''),
                                    'split': encounter_metadata.get('split', 'train')
                                }
                                samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def _get_dermavqa_image_path(self, encounter_id: str, image_id: str, dermavqa_path: Path) -> str:
        """Get DermAVQA image path"""
        # Try different possible paths
        possible_paths = [
            dermavqa_path / 'data' / 'iiyi' / 'images_final' / 'images_train' / image_id,
            dermavqa_path / 'data' / 'iiyi' / 'images_final' / 'images_valid' / image_id,
            dermavqa_path / 'data' / 'iiyi' / 'images_final' / 'images_test' / image_id,
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_skin_tone_description(self, skin_tone_value):
        """Convert numeric skin tone to descriptive text"""
        if pd.isna(skin_tone_value):
            return None
        if skin_tone_value <= 20:
            return "very light"
        elif skin_tone_value <= 40:
            return "light"
        elif skin_tone_value <= 60:
            return "medium"
        elif skin_tone_value <= 80:
            return "medium-dark"
        else:
            return "dark"
    
    def get_fitzpatrick_description(self, scale):
        """Convert Fitzpatrick scale to descriptive text"""
        descriptions = {
            1: "very light (Type I)",
            2: "light (Type II)",
            3: "medium-light (Type III)",
            4: "medium (Type IV)",
            5: "medium-dark (Type V)",
            6: "dark (Type VI)"
        }
        return descriptions.get(scale, None)
    
    def get_body_part(self, row):
        """Extract body part from SCIN data"""
        body_parts = []
        for part in ['body_parts_head_or_neck', 'body_parts_arm', 'body_parts_palm', 'body_parts_back_of_hand', 
                    'body_parts_torso_front', 'body_parts_torso_back', 'body_parts_genitalia_or_groin', 
                    'body_parts_buttocks', 'body_parts_leg', 'body_parts_foot_top_or_side', 'body_parts_foot_sole']:
            if part in row and row[part] == 'YES':
                body_parts.append(part.replace('body_parts_', '').replace('_', ' '))
        return ', '.join(body_parts) if body_parts else ''
    
    def get_symptoms(self, row):
        """Extract symptoms from SCIN data"""
        symptoms = []
        for symptom in ['condition_symptoms_bothersome_appearance', 'condition_symptoms_bleeding', 
                       'condition_symptoms_increasing_size', 'condition_symptoms_darkening', 
                       'condition_symptoms_itching', 'condition_symptoms_burning', 'condition_symptoms_pain']:
            if symptom in row and row[symptom] == 'YES':
                symptoms.append(symptom.replace('condition_symptoms_', '').replace('_', ' '))
        return ', '.join(symptoms) if symptoms else ''
    
    def get_morphological_features(self, row):
        """Extract morphological features from SkinCAP data"""
        morphological_features = []
        feature_columns = ['Vesicle', 'Papule', 'Macule', 'Plaque', 'Abscess', 'Pustule', 'Bulla', 'Patch', 
                          'Nodule', 'Ulcer', 'Crust', 'Erosion', 'Excoriation', 'Atrophy', 'Exudate', 
                          'Purpura/Petechiae', 'Fissure', 'Induration', 'Xerosis', 'Telangiectasia', 'Scale', 
                          'Scar', 'Friable', 'Sclerosis', 'Pedunculated', 'Exophytic/Fungating', 
                          'Warty/Papillomatous', 'Dome-shaped', 'Flat topped', 'Brown(Hyperpigmentation)', 
                          'Translucent', 'White(Hypopigmentation)', 'Purple', 'Yellow', 'Black', 'Erythema', 
                          'Comedo', 'Lichenification', 'Blue', 'Umbilicated', 'Poikiloderma', 'Salmon', 'Wheal', 
                          'Acuminate', 'Burrow', 'Gray', 'Pigmented', 'Cyst']
        
        for feature in feature_columns:
            if feature in row and row[feature] == 1:
                morphological_features.append(feature.lower().replace('(', '').replace(')', ''))
        
        return morphological_features[:5]  # Limit to top 5 features
    
    def generate_model_response(self, disease, metadata, dataset_name):
        """Generate model response based on available metadata"""
        response = f"Based on the image and your description, this appears to be {disease}. Here's what I can observe:\n\n"
        
        # Visual assessment (model should learn to detect)
        visual_features = []
        
        if 'skin_tone' in metadata and metadata['skin_tone']:
            skin_tone_desc = self.get_skin_tone_description(metadata['skin_tone'])
            visual_features.append(f"skin tone: {skin_tone_desc}")
        
        if 'morphological_features' in metadata and metadata['morphological_features']:
            visual_features.append(f"morphology: {', '.join(metadata['morphological_features'])}")
        
        if 'malignant' in metadata and metadata['malignant'] is not None:
            malignancy = "malignant" if metadata['malignant'] else "benign"
            visual_features.append(f"malignancy risk: {malignancy}")
        
        if visual_features:
            response += f"**Visual Assessment:**\n- {', '.join(visual_features)}\n\n"
        
        # Clinical context (from user input)
        clinical_context = []
        if 'age_group' in metadata and metadata['age_group']:
            clinical_context.append(f"Age: {metadata['age_group']}")
        if 'sex' in metadata and metadata['sex']:
            clinical_context.append(f"Gender: {metadata['sex']}")
        if 'duration' in metadata and metadata['duration']:
            clinical_context.append(f"Duration: {metadata['duration']}")
        if 'body_part' in metadata and metadata['body_part']:
            clinical_context.append(f"Location: {metadata['body_part']}")
        if 'symptoms' in metadata and metadata['symptoms']:
            clinical_context.append(f"Symptoms: {metadata['symptoms']}")
        
        if clinical_context:
            response += f"**Clinical Context:**\n- {', '.join(clinical_context)}\n\n"
        
        # Assessment and recommendations
        malignancy_status = metadata.get('malignant', False)
        response += f"**Assessment:** This appears to be a {'malignant' if malignancy_status else 'benign'} condition. "
        
        if malignancy_status:
            response += "This requires immediate medical evaluation and should be seen by a dermatologist as soon as possible."
        else:
            response += "While this may not require immediate medical attention, I recommend consulting a dermatologist for proper evaluation and treatment."
        
        return response
    
    def generate_follow_up_response(self, question, previous_response, disease):
        """Generate appropriate follow-up responses"""
        question_lower = question.lower()
        
        if "more about" in question_lower:
            return f"This condition typically affects various age groups and is characterized by specific features. It's important to monitor for any changes and consult a healthcare provider if symptoms worsen."
        elif "do next" in question_lower:
            return "I recommend monitoring the condition for any changes. If it worsens, changes color, or becomes painful, please consult a dermatologist immediately."
        elif "serious" in question_lower:
            return "While this condition may not be immediately serious, it's important to have it evaluated by a healthcare professional to rule out any concerning features."
        elif "doctor" in question_lower:
            return "Yes, I recommend consulting a dermatologist for proper evaluation and treatment. Early detection and treatment are important for skin conditions."
        elif "causes" in question_lower:
            return "This condition can be caused by various factors including genetics, environmental factors, and lifestyle. A dermatologist can provide more specific information about the underlying causes."
        elif "spread" in question_lower:
            return "The spread of this condition depends on the specific type. Some conditions can spread to other areas, while others remain localized. A dermatologist can provide specific information about your case."
        elif "last" in question_lower:
            return "The duration varies greatly depending on the condition and individual factors. Some conditions resolve quickly, while others may be chronic. Proper treatment can help manage symptoms."
        elif "treatments" in question_lower:
            return "Treatment options vary depending on the specific condition and severity. A dermatologist can recommend appropriate treatments, which may include topical medications, oral medications, or other therapies."
        else:
            return "I'd be happy to provide more information. Please consult with a healthcare professional for personalized advice and treatment options."
    
    def process_dermavqa_sample(self, row):
        """Process DermAVQA sample with multilingual conversation data"""
        # Extract available metadata
        encounter_id = row.get('encounter_id', 'unknown')
        age = row.get('age', None)
        sex = row.get('sex', None)
        anatomical_location = row.get('anatomical_location', row.get('anatomic_locations', ''))
        validation_level = row.get('validation_level', '')
        completeness = row.get('completeness', 0)
        
        # Create multilingual conversations
        conversations = []
        languages = [
            ('en', 'query_title_en', 'query_content_en', 'response_en'),
            ('zh', 'query_title_zh', 'query_content_zh', 'response_zh'),
            ('es', 'query_title_es', 'query_content_es', 'response_es')
        ]
        
        for lang_code, title_col, content_col, response_col in languages:
            query_title = row.get(title_col, '')
            query_content = row.get(content_col, '')
            response_content = row.get(response_col, '')
            
            # Skip if no content available for this language
            if not query_title and not query_content and not response_content:
                continue
            
            # Generate user input from query
            if query_title and query_content:
                user_input = f"{query_title}\n\n{query_content}"
            elif query_title:
                user_input = query_title
            elif query_content:
                user_input = query_content
            else:
                user_input = "Can you help me identify this skin condition?"
            
            # Generate enhanced model response with clinical context
            enhanced_response = f"{response_content}\n\n**Clinical Context:**\n"
            if age:
                enhanced_response += f"- Age: {age} years old\n"
            if sex:
                enhanced_response += f"- Gender: {sex}\n"
            if anatomical_location:
                enhanced_response += f"- Location: {anatomical_location}\n"
            if validation_level:
                enhanced_response += f"- Response quality: {validation_level} (completeness: {completeness})\n"
            
            enhanced_response += f"\n**Note:** This response was provided by a validated medical professional."
            
            # Create conversation for this language
            conversation = {
                "image": row['image_path'],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{user_input}"},
                    {"from": "gpt", "value": enhanced_response}
                ],
                "metadata": {
                    "dataset": "dermavqa",
                    "language": lang_code,
                    "encounter_id": encounter_id,
                    "age": age,
                    "sex": sex,
                    "anatomical_location": anatomical_location,
                    "validation_level": validation_level,
                    "completeness": completeness,
                    "multilingual": {
                        "query_zh": row.get('query_title_zh', '') + ' ' + row.get('query_content_zh', ''),
                        "query_en": row.get('query_title_en', '') + ' ' + row.get('query_content_en', ''),
                        "query_es": row.get('query_title_es', '') + ' ' + row.get('query_content_es', ''),
                        "response_zh": row.get('response_zh', ''),
                        "response_en": row.get('response_en', ''),
                        "response_es": row.get('response_es', '')
                    }
                }
            }
            conversations.append(conversation)
        
        # Return list of conversations (one per language)
        return conversations
    
    def process_ddi_sample(self, row):
        """Process DDI sample with synthetic conversation generation"""
        # Extract available metadata
        skin_tone = row.get('skin_tone', None)
        malignant = row.get('malignant', None)
        disease = row['disease']
        
        # Generate synthetic user input
        user_input = f"I have this skin condition. It's been there for a while. What could this be?"
        
        # Generate model response with visual assessment
        metadata = {
            'skin_tone': skin_tone,
            'malignant': malignant
        }
        
        model_response = self.generate_model_response(disease, metadata, 'ddi')
        
        return {
            "image": row['image_path'],
            "conversations": [
                {"from": "human", "value": f"<image>\n{user_input}"},
                {"from": "gpt", "value": model_response}
            ],
            "metadata": {
                "dataset": "ddi",
                "skin_tone": skin_tone,
                "malignant": malignant,
                "disease": disease
            }
        }
    
    def process_scin_sample(self, row):
        """Process SCIN sample with synthetic conversation generation"""
        # Extract user-provided information and normalize unknowns
        raw_age_group = row.get('age_group', None)
        raw_sex = row.get('sex_at_birth', None)
        raw_body_part = self.get_body_part(row)
        raw_duration = row.get('condition_duration', None)
        raw_symptoms = self.get_symptoms(row)
        raw_disease = row.get('most_confident_label', None)

        age_group = None if self._is_useless_value(raw_age_group) else str(raw_age_group)
        sex = None if self._is_useless_value(raw_sex) else str(raw_sex)
        body_part = None if self._is_useless_value(raw_body_part) else str(raw_body_part)
        duration = None if self._is_useless_value(raw_duration) else str(raw_duration)
        symptoms = None if self._is_useless_value(raw_symptoms) else str(raw_symptoms)
        disease = None if self._is_useless_value(raw_disease) else str(raw_disease)
        if not disease:
            disease = 'condition'
        
        # Compose user input with only available pieces
        parts = ["I"]
        if duration:
            parts.append(f"noticed this {duration.replace('_', ' ').lower()} ago")
        else:
            parts.append("have this skin condition")
        if body_part:
            parts.append(f"on my {body_part}")
        sentence1 = " ".join(parts).strip() + "."

        demo_parts = []
        if age_group:
            demo_parts.append(str(age_group).replace('AGE_', '').replace('_', ' ').lower())
        if sex:
            demo_parts.append(str(sex).replace('_', ' ').lower())
        sentence2 = f"I'm a {' '.join(demo_parts)}." if demo_parts else ""

        sentence3 = f"It's {symptoms}." if symptoms else ""
        sentence4 = "What could this be?"
        user_input = " ".join([s for s in [sentence1, sentence2, sentence3, sentence4] if s]).strip()
        
        # Generate model response
        metadata = {
            'age_group': age_group,
            'sex': sex,
            'body_part': body_part,
            'duration': duration,
            'symptoms': symptoms
        }
        
        model_response = self.generate_model_response(disease, metadata, 'scin')
        
        return {
            "image": row['image_path'],  # Use image_path from unified dataset
            "conversations": [
                {"from": "human", "value": f"<image>\n{user_input}"},
                {"from": "gpt", "value": self._sanitize_text(model_response)}
            ],
            "metadata": {k: v for k, v in {
                "dataset": "scin",
                "age_group": age_group,
                "sex": sex,
                "body_part": body_part,
                "duration": duration,
                "symptoms": symptoms,
                "disease": disease
            }.items() if not self._is_useless_value(v)}
        }
    
    def process_skincap_sample(self, row):
        """Process SkinCAP sample with synthetic conversation generation"""
        # Extract morphological features
        morphological_features = self.get_morphological_features(row)
        malignant = row.get('malignant', None)
        disease = row['disease']
        
        # Generate user input
        user_input = f"I have this skin condition. What could this be?"
        
        # Generate model response with morphological analysis
        metadata = {
            'morphological_features': morphological_features,
            'malignant': malignant
        }
        
        model_response = f"Based on the image, this appears to be {disease}. "
        if morphological_features:
            model_response += f"I can observe the following morphological features: {', '.join(morphological_features)}. "
        model_response += f"This appears to be a {'malignant' if malignant else 'benign'} condition. "
        model_response += f"{'This requires immediate medical evaluation' if malignant else 'This may not require immediate medical attention'}."
        
        return {
            "image": row['image_path'],  # Use image_path from unified dataset
            "conversations": [
                {"from": "human", "value": f"<image>\n{user_input}"},
                {"from": "gpt", "value": model_response}
            ],
            "metadata": {
                "dataset": "skincap",
                "morphological_features": morphological_features,
                "malignant": malignant,
                "disease": disease
            }
        }
    
    def process_fitzpatrick_sample(self, row):
        """Process Fitzpatrick17k sample with synthetic conversation generation"""
        # Extract available metadata
        fitzpatrick_scale = row.get('fitzpatrick_scale', None)
        nine_partition = row.get('nine_partition_label', 'unknown')
        three_partition = row.get('three_partition_label', 'unknown')
        disease = row['disease']  # Use 'disease' from unified dataset
        
        # Generate user input
        user_input = f"I have this skin condition. What could this be?"
        
        # Generate model response
        metadata = {
            'fitzpatrick_scale': fitzpatrick_scale,
            'nine_partition': nine_partition,
            'three_partition': three_partition
        }
        
        model_response = f"Based on the image, this appears to be {disease}. "
        if fitzpatrick_scale:
            model_response += f"I can observe this is on {self.get_fitzpatrick_description(fitzpatrick_scale)} skin. "
        model_response += f"This is classified as a {nine_partition} condition ({three_partition}). "
        model_response += f"This appears to be a {'malignant' if 'malignant' in three_partition else 'benign'} condition."
        
        return {
            "image": row['image_path'],  # Use image_path from unified dataset
            "conversations": [
                {"from": "human", "value": f"<image>\n{user_input}"},
                {"from": "gpt", "value": model_response}
            ],
            "metadata": {
                "dataset": "fitzpatrick",
                "fitzpatrick_scale": fitzpatrick_scale,
                "nine_partition": nine_partition,
                "three_partition": three_partition,
                "disease": disease
            }
        }
    
    def enhance_conversation_diversity(self, conversation):
        """Add follow-up questions to enhance conversation diversity"""
        # Randomly add follow-up (30% chance)
        if random.random() < 0.3:
            follow_up = random.choice(self.follow_up_templates)
            disease = conversation['metadata'].get('disease', 'this condition')
            
            conversation["conversations"].append({
                "from": "human",
                "value": follow_up
            })
            conversation["conversations"].append({
                "from": "gpt",
                "value": self.generate_follow_up_response(follow_up, conversation["conversations"][1]["value"], disease)
            })
        
        return conversation
    
    def create_synthetic_conversations(self, dataset_name, samples):
        """Create synthetic conversations for a dataset"""
        conversations = []
        
        for _, sample in samples.iterrows():
            if dataset_name == "ddi":
                conv = self.process_ddi_sample(sample)
                conversations.append(conv)
            elif dataset_name == "scin":
                conv = self.process_scin_sample(sample)
                conversations.append(conv)
            elif dataset_name == "skincap":
                conv = self.process_skincap_sample(sample)
                conversations.append(conv)
            elif dataset_name == "fitzpatrick":
                conv = self.process_fitzpatrick_sample(sample)
                conversations.append(conv)
            elif dataset_name == "dermavqa":
                convs = self.process_dermavqa_sample(sample)
                # DermAVQA returns a list of conversations (one per language)
                for conv in convs:
                    # Enhance with diversity
                    conv = self.enhance_conversation_diversity(conv)
                    conversations.append(conv)
            else:
                # Fallback for other datasets
                conv = {
                    "image": sample.get('image_path', sample.get('image', 'unknown')),
                    "conversations": [
                        {"from": "human", "value": f"<image>\nWhat skin condition is shown in this image?"},
                        {"from": "gpt", "value": f"This image shows {sample.get('disease', 'unknown condition')}."}
                    ],
                    "metadata": {"dataset": dataset_name, "disease": sample.get('disease', 'unknown')}
                }
                
                # Enhance with diversity
                conv = self.enhance_conversation_diversity(conv)
                conversations.append(conv)
        
        return conversations
    
    def analyze_multilingual_distribution(self, train_data, val_data):
        """Analyze the distribution of languages in the training data"""
        print(f"\n" + "="*80)
        print("MULTILINGUAL DATA DISTRIBUTION ANALYSIS")
        print("="*80)
        
        all_data = train_data + val_data
        
        # Count languages
        language_counts = {}
        dataset_language_counts = {}
        
        for item in all_data:
            dataset = item['metadata'].get('dataset', 'unknown')
            language = item['metadata'].get('language', 'en')  # Default to English for non-DermAVQA
            
            if dataset not in dataset_language_counts:
                dataset_language_counts[dataset] = {}
            
            if language not in dataset_language_counts[dataset]:
                dataset_language_counts[dataset][language] = 0
            
            dataset_language_counts[dataset][language] += 1
            language_counts[language] = language_counts.get(language, 0) + 1
        
        print(f"Total training samples: {len(all_data):,}")
        print(f"\nLanguage distribution:")
        for lang, count in sorted(language_counts.items()):
            percentage = count / len(all_data) * 100
            print(f"  {lang.upper()}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\nLanguage distribution by dataset:")
        for dataset, lang_counts in dataset_language_counts.items():
            print(f"  {dataset.upper()}:")
            for lang, count in sorted(lang_counts.items()):
                percentage = count / sum(lang_counts.values()) * 100
                print(f"    {lang.upper()}: {count:,} samples ({percentage:.1f}%)")
        
        # DermAVQA specific analysis
        dermavqa_samples = [item for item in all_data if item['metadata'].get('dataset') == 'dermavqa']
        if dermavqa_samples:
            print(f"\nDermAVQA Multilingual Analysis:")
            print(f"  Total DermAVQA samples: {len(dermavqa_samples):,}")
            
            # Analyze validation levels
            validation_levels = {}
            completeness_scores = {}
            
            for item in dermavqa_samples:
                validation_level = item['metadata'].get('validation_level', 'unknown')
                completeness = item['metadata'].get('completeness', 0)
                
                validation_levels[validation_level] = validation_levels.get(validation_level, 0) + 1
                completeness_scores[completeness] = completeness_scores.get(completeness, 0) + 1
            
            print(f"  Validation levels: {validation_levels}")
            print(f"  Completeness scores: {completeness_scores}")
    
    def analyze_dataset_quality(self):
        """Analyze dataset quality and provide recommendations"""
        print("\n" + "="*80)
        print("DATASET QUALITY ANALYSIS")
        print("="*80)
        
        for dataset_name, df in self.datasets.items():
            print(f"\n📊 {dataset_name.upper()} Dataset Analysis:")
            
            if dataset_name == 'scin':
                disease_counts = df['most_confident_label'].value_counts()
                disease_col = 'most_confident_label'
            elif dataset_name == 'dermnet':
                disease_counts = df['disease'].value_counts()
                disease_col = 'disease'
            elif dataset_name == 'dermavqa':
                # DermAVQA doesn't have explicit disease labels, skip disease analysis
                print(f"  Total samples: {len(df):,}")
                print(f"  Dataset type: Real conversations with medical validation")
                print(f"  Languages: Chinese, English, Spanish")
                if not df.empty and 'validation_level' in df.columns:
                    print(f"  Quality validation: {df['validation_level'].value_counts().to_dict()}")
                if not df.empty and 'completeness' in df.columns:
                    print(f"  Completeness scores: {df['completeness'].value_counts().to_dict()}")
                continue
            else:
                disease_col = 'disease' if 'disease' in df.columns else 'label'
                disease_counts = df[disease_col].value_counts()
            
            total_samples = len(df)
            total_diseases = len(disease_counts)
            
            # Quality metrics
            single_example = len(disease_counts[disease_counts == 1])
            few_examples = len(disease_counts[(disease_counts >= 2) & (disease_counts <= 5)])
            moderate_examples = len(disease_counts[(disease_counts >= 6) & (disease_counts <= 20)])
            adequate_examples = len(disease_counts[(disease_counts >= 21) & (disease_counts <= 50)])
            good_examples = len(disease_counts[disease_counts > 50])
            
            print(f"  Total samples: {total_samples:,}")
            print(f"  Total diseases: {total_diseases}")
            print(f"  Average samples per disease: {total_samples/total_diseases:.1f}")
            print(f"  ❌ Single example (remove): {single_example} ({single_example/total_diseases*100:.1f}%)")
            print(f"  ⚠️  Few examples (2-5): {few_examples} ({few_examples/total_diseases*100:.1f}%)")
            print(f"  ⚠️  Moderate examples (6-20): {moderate_examples} ({moderate_examples/total_diseases*100:.1f}%)")
            print(f"  ✅ Adequate examples (21-50): {adequate_examples} ({adequate_examples/total_diseases*100:.1f}%)")
            print(f"  ✅ Good examples (50+): {good_examples} ({good_examples/total_diseases*100:.1f}%)")
            
            # Training suitability
            trainable_diseases = len(disease_counts[disease_counts >= self.min_samples_threshold])
            trainable_samples = disease_counts[disease_counts >= self.min_samples_threshold].sum()
            
            print(f"  🎯 Trainable diseases (≥{self.min_samples_threshold}): {trainable_diseases}/{total_diseases} ({trainable_diseases/total_diseases*100:.1f}%)")
            print(f"  🎯 Trainable samples: {trainable_samples:,}/{total_samples:,} ({trainable_samples/total_samples*100:.1f}%)")
    
    def create_unified_dataset(self, min_samples: int = 1):
        """Create unified dataset with quality filtering"""
        print(f"\n" + "="*80)
        print(f"CREATING UNIFIED DATASET (min_samples={min_samples})")
        print("="*80)
        
        unified_samples = []
        
        for dataset_name, df in self.datasets.items():
            print(f"\nProcessing {dataset_name}...")
            
            if dataset_name == 'scin':
                disease_counts = df['most_confident_label'].value_counts()
                disease_col = 'most_confident_label'
                image_col = 'case_id'  # SCIN uses case_id for image reference
            elif dataset_name == 'dermnet':
                disease_counts = df['disease'].value_counts()
                disease_col = 'disease'
                image_col = 'image_path'
            elif dataset_name == 'dermavqa':
                # DermAVQA doesn't have disease labels, add all samples
                df['disease'] = 'dermavqa_conversation'
                disease_counts = df['disease'].value_counts()
                disease_col = 'disease'
                image_col = 'image_path'
            elif dataset_name == 'skincap':
                disease_counts = df['disease'].value_counts()
                disease_col = 'disease'
                image_col = 'skincap_file_path'  # Local CSV has skincap_file_path column
            else:
                disease_col = 'disease' if 'disease' in df.columns else 'label'
                disease_counts = df[disease_col].value_counts()
                image_col = 'image' if 'image' in df.columns else 'image_path'
            
            # Filter diseases with sufficient samples
            trainable_diseases = disease_counts[disease_counts >= min_samples].index
            filtered_df = df[df[disease_col].isin(trainable_diseases)]
            
            print(f"  Original: {len(df)} samples, {len(disease_counts)} diseases")
            print(f"  Filtered: {len(filtered_df)} samples, {len(trainable_diseases)} diseases")
            
            # Convert to unified format
            for _, row in filtered_df.iterrows():
                if dataset_name == 'scin':
                    # SCIN has image_1_path column with correct image paths
                    # Extract filename from the full path
                    image_filename = Path(row['image_1_path']).name
                    image_path = str(self.data_root / 'scin' / 'dataset' / 'images' / image_filename)
                    if os.path.exists(image_path):
                        unified_samples.append({
                            'image_path': image_path,
                            'disease': row[disease_col],
                            'dataset': dataset_name,
                            'split': 'train'  # SCIN doesn't have explicit splits
                        })
                elif dataset_name == 'dermnet':
                    unified_samples.append({
                        'image_path': row['image_path'],
                        'disease': row[disease_col],
                        'dataset': dataset_name,
                        'split': row['split']
                    })
                elif dataset_name == 'dermavqa':
                    unified_samples.append({
                        'image_path': row['image_path'],
                        'disease': 'dermavqa_conversation',
                        'dataset': dataset_name,
                        'split': row.get('split', 'train')
                    })
                elif dataset_name == 'skincap':
                    # SkinCap references DDI images via ori_file_path
                    ddi_image_path = str(self.data_root / 'ddidiversedermatologyimages' / row['ori_file_path'])
                    if os.path.exists(ddi_image_path):
                        unified_samples.append({
                            'image_path': ddi_image_path,
                            'disease': row[disease_col],
                            'dataset': dataset_name,
                            'split': 'train'  # SkinCap doesn't have explicit splits
                        })
                elif dataset_name == 'ddi':
                    # DDI has DDI_file column with PNG filenames
                    image_path = str(self.dataset_paths['ddi'] / row['DDI_file'])
                    if os.path.exists(image_path):
                        unified_samples.append({
                            'image_path': image_path,
                            'disease': row[disease_col],
                            'dataset': dataset_name,
                            'split': 'train'  # DDI doesn't have explicit splits
                        })
                elif dataset_name == 'fitzpatrick':
                    # Fitzpatrick17k has URLs - use URL as image path for now
                    # In training, we'll need to handle URL loading
                    unified_samples.append({
                        'image_path': row['url'],  # Use URL as image path
                        'disease': row[disease_col],
                        'dataset': dataset_name,
                        'split': 'train'  # Fitzpatrick17k doesn't have explicit splits
                    })
            
        
        # Create unified DataFrame
        unified_df = pd.DataFrame(unified_samples)

        # Drop rows with NaN/unknown/useless disease or missing/empty image paths
        if not unified_df.empty:
            unified_df = unified_df.dropna(subset=['disease', 'image_path'])
            unified_df = unified_df[~unified_df['disease'].astype(str).str.strip().str.lower().isin(['unknown', 'unknown condition', ''])]
            unified_df = unified_df[unified_df['image_path'].astype(str).str.strip() != '']
        
        print(f"\n📊 UNIFIED DATASET SUMMARY:")
        print(f"Total samples: {len(unified_df):,}")
        print(f"Total diseases: {unified_df['disease'].nunique()}")
        print(f"Average samples per disease: {len(unified_df)/unified_df['disease'].nunique():.1f}")
        
        # Dataset distribution
        print(f"\nDataset distribution:")
        for dataset in unified_df['dataset'].unique():
            count = len(unified_df[unified_df['dataset'] == dataset])
            print(f"  {dataset}: {count:,} samples ({count/len(unified_df)*100:.1f}%)")
        
        # Disease distribution
        disease_counts = unified_df['disease'].value_counts()
        print(f"\nTop 10 diseases:")
        for i, (disease, count) in enumerate(disease_counts.head(10).items(), 1):
            print(f"  {i:2d}. {disease}: {count} samples")
        
        # Save unified dataset for debugging
        unified_df.to_csv('stage1_data/unified_dataset.csv', index=False)
        print(f"\n💾 Unified dataset saved to: stage1_data/unified_dataset.csv")
        
        return unified_df
    
    def _get_scin_image_path(self, case_id: str) -> str:
        """Get SCIN image path from case_id"""
        # SCIN images are stored in dataset/images/ directory
        possible_paths = [
            self.data_root / 'scin' / 'dataset' / 'images' / f"{case_id}.jpg",
            self.data_root / 'scin' / 'dataset' / 'images' / f"{case_id}.png",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        return None
    
    def _get_image_path(self, row: pd.Series, dataset_name: str) -> str:
        """Get image path for DDI and Fitzpatrick17k datasets"""
        if dataset_name == 'ddi':
            # DDI has DDI_file column with PNG filenames
            if 'DDI_file' in row:
                return str(self.dataset_paths['ddi'] / row['DDI_file'])
        elif dataset_name == 'fitzpatrick':
            # Fitzpatrick17k only has URLs, no local images - skip
            return None
        return None
    
    def create_training_data(self, unified_df: pd.DataFrame, train_ratio: float = 0.8):
        """Create training data in LLaVA format"""
        print(f"\n" + "="*80)
        print("CREATING TRAINING DATA IN LLAVA FORMAT")
        print("="*80)
        
        # Create train/val split
        train_samples = []
        val_samples = []
        
        # Stratified split by disease
        for disease in unified_df['disease'].unique():
            disease_samples = unified_df[unified_df['disease'] == disease]
            
            # Shuffle and split
            disease_samples = disease_samples.sample(frac=1, random_state=42)
            split_idx = int(len(disease_samples) * train_ratio)
            
            train_samples.append(disease_samples.iloc[:split_idx])
            val_samples.append(disease_samples.iloc[split_idx:])
        
        train_df = pd.concat(train_samples, ignore_index=True)
        val_df = pd.concat(val_samples, ignore_index=True)
        
        print(f"Train samples: {len(train_df):,}")
        print(f"Val samples: {len(val_df):,}")
        
        # Create LLaVA format data with synthetic conversations
        train_data = self._create_llava_format_with_synthetic_conversations(train_df)
        val_data = self._create_llava_format_with_synthetic_conversations(val_df)
        
        # Save training data
        train_file = self.output_dir / "train.jsonl"
        val_file = self.output_dir / "val.jsonl"
        
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        with open(val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"\n✓ Training data saved to: {train_file}")
        print(f"✓ Validation data saved to: {val_file}")
        
        # Create metadata
        metadata = {
            'total_train_samples': len(train_data),
            'total_val_samples': len(val_data),
            'total_diseases': unified_df['disease'].nunique(),
            'datasets_used': list(unified_df['dataset'].unique()),
            'min_samples_threshold': self.min_samples_threshold,
            'disease_distribution': unified_df['disease'].value_counts().to_dict()
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {self.output_dir / 'metadata.json'}")
        
        return train_data, val_data
    
    def _create_llava_format_with_synthetic_conversations(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to LLaVA format with synthetic conversations"""
        print(f"Creating LLaVA format for {len(df)} samples...")
        
        data = []
        
        # Group by dataset for synthetic conversation generation
        for dataset_name in df['dataset'].unique():
            dataset_samples = df[df['dataset'] == dataset_name]
            print(f"Processing {dataset_name}: {len(dataset_samples)} samples")
            
            # Create synthetic conversations for this dataset
            conversations = self.create_synthetic_conversations(dataset_name, dataset_samples)
            # Sanitize generated conversations
            cleaned_conversations = []
            for item in conversations:
                cleaned = self._sanitize_item(item)
                if cleaned is not None:
                    cleaned_conversations.append(cleaned)
            conversations = cleaned_conversations
            print(f"Generated {len(conversations)} conversations for {dataset_name}")
            
            data.extend(conversations)
        
        print(f"Total training samples created: {len(data)}")
        return data
    
    def _create_llava_format(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to LLaVA format"""
        data = []
        
        for _, row in df.iterrows():
            # Create conversation format
            conversation = [
                {
                    "from": "human",
                    "value": "<image>\nWhat skin condition is shown in this image?"
                },
                {
                    "from": "gpt", 
                    "value": f"This image shows {row['disease']}."
                }
            ]
            
            data.append({
                "image": row['image_path'],
                "conversations": conversation
            })
        
        return data
    
    def validate_data(self, train_file: str, val_file: str):
        """Validate the created training data"""
        print(f"\n" + "="*80)
        print("VALIDATING TRAINING DATA")
        print("="*80)
        
        # Load and validate train data
        train_data = []
        with open(train_file, 'r') as f:
            for line in f:
                train_data.append(json.loads(line))
        
        val_data = []
        with open(val_file, 'r') as f:
            for line in f:
                val_data.append(json.loads(line))
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        # Check image paths (skip URLs and non-string paths)
        missing_images = 0
        for data in train_data + val_data:
            img = data.get('image')
            if not isinstance(img, str) or img.startswith('http://') or img.startswith('https://'):
                continue
            if not os.path.exists(img):
                missing_images += 1
        
        print(f"Missing images: {missing_images}")
        
        # Check conversation format
        invalid_conversations = 0
        for data in train_data + val_data:
            if 'conversations' not in data or len(data['conversations']) != 2:
                invalid_conversations += 1
        
        print(f"Invalid conversations: {invalid_conversations}")
        
        if missing_images == 0 and invalid_conversations == 0:
            print("✅ All data validation checks passed!")
        else:
            print("❌ Some validation checks failed!")
        
        return missing_images == 0 and invalid_conversations == 0

def main():
    """Main function to run the data preparation pipeline"""
    print("Stage 1 Data Preparation Pipeline for Dermatology Domain Adaptation")
    print("="*80)
    
    # Initialize processor
    processor = DermatologyDataProcessor()
    
    # Analyze dataset quality
    processor.analyze_dataset_quality()
    
    # Create unified dataset with quality filtering
    unified_df = processor.create_unified_dataset(min_samples=1)
    
    # Create training data
    train_data, val_data = processor.create_training_data(unified_df)
    
    # Analyze multilingual distribution
    processor.analyze_multilingual_distribution(train_data, val_data)
    
    # Validate data
    train_file = processor.output_dir / "train.jsonl"
    val_file = processor.output_dir / "val.jsonl"
    processor.validate_data(train_file, val_file)
    
    print(f"\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)
    print(f"Output directory: {processor.output_dir}")
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")
    print(f"Metadata: {processor.output_dir / 'metadata.json'}")
    print("\nNext steps:")
    print("1. Review the generated synthetic conversations")
    print("2. Run Stage 1 training with the prepared data")
    print("3. Monitor training progress and adjust parameters as needed")
    print("\nSynthetic Conversation Features:")
    print("- Realistic patient-doctor interactions")
    print("- Metadata integration (skin tone, demographics, symptoms)")
    print("- Multi-turn conversations with follow-up questions")
    print("- Clinical reasoning and safety recommendations")
    print("- Multilingual training data (English, Chinese, Spanish)")
    print("- Real conversations from DermAVQA with medical validation")

if __name__ == "__main__":
    main()
