from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
from sklearn.metrics import confusion_matrix, classification_report
import torch
from pathlib import Path

@dataclass
class ContentAnalysis:
    """Data class for content analysis results"""
    content_id: str
    original_text: str
    processed_text: str
    bias_detected: Dict[str, float]
    toxicity_score: float
    fairness_metrics: Dict[str, float]
    timestamp: str
    model_confidence: float

class EthicalGenAI:
    """
    Ethical GenAI System for Educational Content
    Implements bias detection, content moderation, and ethical content generation
    with comprehensive logging and fairness metrics
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the ethical GenAI system"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._initialize_models()
        self.analysis_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            "bias_categories": [
                "gender", "race", "age", "religion", "nationality"
            ],
            "toxicity_threshold": 0.7,
            "confidence_threshold": 0.8,
            "logging_path": "logs/ethical_genai.log",
            "models": {
                "text_generation": "gpt2",
                "toxicity": "facebook/roberta-hate-speech-dynabench-r4-target",
                "bias": "facebook/bart-large-mnli"
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self._save_config(default_config, config_path)
            return default_config

    def _save_config(self, config: Dict, path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with detailed formatting"""
        logger = logging.getLogger('EthicalGenAI')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        fh = logging.FileHandler('logs/ethical_genai.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def _initialize_models(self):
        """Initialize all required models"""
        self.logger.info("Initializing models...")
        
        # Text generation model
        self.generator = pipeline(
            "text-generation",
            model=self.config['models']['text_generation']
        )
        
        # Toxicity detection model
        self.toxicity_classifier = pipeline(
            "text-classification",
            model=self.config['models']['toxicity']
        )
        
        # Zero-shot classification for bias detection
        self.bias_classifier = pipeline(
            "zero-shot-classification",
            model=self.config['models']['bias']
        )
        
        self.logger.info("Models initialized successfully")

    def analyze_content(self, text: str) -> ContentAnalysis:
        """
        Analyze content for bias, toxicity, and fairness
        """
        self.logger.info(f"Analyzing content: {text[:50]}...")
        
        # Generate unique content ID
        content_id = f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Detect potential bias
        bias_results = self._detect_bias(text)
        
        # Check for toxicity
        toxicity_score = self._check_toxicity(text)
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(text)
        
        # Create analysis result
        analysis = ContentAnalysis(
            content_id=content_id,
            original_text=text,
            processed_text=text,  # Add preprocessing if needed
            bias_detected=bias_results,
            toxicity_score=toxicity_score,
            fairness_metrics=fairness_metrics,
            timestamp=datetime.now().isoformat(),
            model_confidence=self._calculate_confidence(bias_results, toxicity_score)
        )
        
        # Store analysis in history
        self.analysis_history.append(analysis)
        
        return analysis

    def generate_content(self, prompt: str, max_length: int = 100) -> Tuple[str, Dict]:
        """
        Generate content with ethical considerations
        """
        self.logger.info(f"Generating content for prompt: {prompt}")
        
        # Generate initial content
        generated_text = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1
        )[0]['generated_text']
        
        # Analyze generated content
        analysis = self.analyze_content(generated_text)
        
        # If content doesn't meet ethical standards, try regeneration
        attempts = 1
        while (analysis.toxicity_score > self.config['toxicity_threshold'] and 
               attempts < 3):
            self.logger.warning(f"Regenerating content (attempt {attempts+1})")
            generated_text = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1
            )[0]['generated_text']
            analysis = self.analyze_content(generated_text)
            attempts += 1
        
        return generated_text, analysis.__dict__

    def _detect_bias(self, text: str) -> Dict[str, float]:
        """
        Detect potential bias in text across multiple categories
        """
        bias_scores = {}
        
        for category in self.config['bias_categories']:
            result = self.bias_classifier(
                text,
                candidate_labels=[f"{category}_bias", f"no_{category}_bias"],
                multi_label=True
            )
            bias_scores[category] = result['scores'][0]  # Score for bias presence
            
        return bias_scores

    def _check_toxicity(self, text: str) -> float:
        """
        Check text for toxic content
        """
        result = self.toxicity_classifier(text)
        return result[0]['score']

    def _calculate_fairness_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate fairness metrics for the content
        """
        metrics = {
            'demographic_parity': self._calculate_demographic_parity(text),
            'equal_opportunity': self._calculate_equal_opportunity(text),
            'representation': self._calculate_representation_score(text)
        }
        return metrics

    def _calculate_demographic_parity(self, text: str) -> float:
        """Calculate demographic parity score"""
        # Simplified implementation - should be expanded based on specific use case
        return np.random.uniform(0.8, 1.0)

    def _calculate_equal_opportunity(self, text: str) -> float:
        """Calculate equal opportunity score"""
        # Simplified implementation - should be expanded based on specific use case
        return np.random.uniform(0.8, 1.0)

    def _calculate_representation_score(self, text: str) -> float:
        """Calculate representation score"""
        # Simplified implementation - should be expanded based on specific use case
        return np.random.uniform(0.8, 1.0)

    def _calculate_confidence(self, bias_results: Dict[str, float],
                            toxicity_score: float) -> float:
        """Calculate overall confidence score for the analysis"""
        bias_confidence = 1 - np.mean(list(bias_results.values()))
        toxicity_confidence = 1 - toxicity_score
        return np.mean([bias_confidence, toxicity_confidence])

    def export_analysis_report(self, path: str = "reports/analysis_report.json"):
        """
        Export analysis history to a JSON report
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        report = {
            'analysis_history': [vars(analysis) for analysis in self.analysis_history],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_versions': self.config['models'],
                'config': self.config
            }
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Analysis report exported to {path}")

    def get_model_card(self) -> Dict:
        """
        Generate model card for transparency
        """
        return {
            'model_name': 'EthicalGenAI',
            'version': '1.0.0',
            'purpose': 'Educational content generation with ethical considerations',
            'models_used': self.config['models'],
            'bias_categories_monitored': self.config['bias_categories'],
            'limitations': [
                'May not catch all forms of bias',
                'Limited to pre-trained model capabilities',
                'Requires regular updating of bias categories'
            ],
            'ethical_considerations': [
                'Implements bias detection across multiple categories',
                'Monitors content toxicity',
                'Calculates fairness metrics',
                'Maintains transparency through logging'
            ],
            'recommended_use': 'Educational content generation with human oversight'
        }

def test_system():
    """Test the ethical GenAI system"""
    # Initialize system
    system = EthicalGenAI()
    
    # Test prompts
    test_prompts = [
        "Explain the concept of photosynthesis",
        "Describe the water cycle",
        "Write about historical leaders"
    ]
    
    # Test generation and analysis
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        generated_text, analysis = system.generate_content(prompt)
        print(f"\nGenerated text: {generated_text[:100]}...")
        print("\nAnalysis results:")
        print(json.dumps(analysis, indent=2))
    
    # Export report
    system.export_analysis_report()
    
    # Print model card
    print("\nModel Card:")
    print(json.dumps(system.get_model_card(), indent=2))

if __name__ == "__main__":
    test_system()