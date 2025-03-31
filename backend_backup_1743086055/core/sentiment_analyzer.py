"""
Sentiment Analysis module for financial text data using FinBERT.

This module provides functionality to analyze sentiment in:
- Financial news
- Earnings reports
- SEC filings

Sentiment is classified as Bullish, Neutral, or Bearish.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A class for analyzing sentiment in financial text using FinBERT.
    """
    
    def __init__(self, model_name="ProsusAI/finbert", device=None):
        """
        Initialize the sentiment analyzer with FinBERT model.
        
        Args:
            model_name (str): Name or path of the FinBERT model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        
        # Set device (GPU if available, otherwise CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing FinBERT sentiment analyzer on {self.device}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise
        
        # Define sentiment labels
        self.labels = ["bearish", "neutral", "bullish"]
        
    def analyze_text(self, text):
        """
        Analyze the sentiment of a single text input.
        
        Args:
            text (str): Financial text to analyze.
            
        Returns:
            dict: Sentiment classification with scores.
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities and label
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
            sentiment_label = self.labels[np.argmax(probabilities)]
            sentiment_score = float(np.max(probabilities))
            
            # Create sentiment dictionary
            sentiment = {
                "label": sentiment_label,
                "score": sentiment_score,
                "probabilities": {
                    label: float(prob) for label, prob in zip(self.labels, probabilities)
                }
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "error", "score": 0.0, "error": str(e)}
    
    def analyze_batch(self, texts):
        """
        Analyze the sentiment of a batch of texts.
        
        Args:
            texts (list): List of financial texts to analyze.
            
        Returns:
            list: List of sentiment dictionaries.
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Create sentiment dictionaries
            sentiments = []
            for i, probs in enumerate(probabilities):
                sentiment_label = self.labels[np.argmax(probs)]
                sentiment_score = float(np.max(probs))
                
                sentiment = {
                    "label": sentiment_label,
                    "score": sentiment_score,
                    "probabilities": {
                        label: float(prob) for label, prob in zip(self.labels, probs)
                    }
                }
                sentiments.append(sentiment)
                
            return sentiments
            
        except Exception as e:
            logger.error(f"Error analyzing batch sentiment: {str(e)}")
            return [{"label": "error", "score": 0.0, "error": str(e)} for _ in range(len(texts))]
            
    def analyze_financial_news(self, news_articles):
        """
        Analyze sentiment in financial news articles.
        
        Args:
            news_articles (list): List of news articles.
            
        Returns:
            list: Sentiment analysis results.
        """
        # Extract text from news articles
        if isinstance(news_articles, list) and all(isinstance(item, dict) for item in news_articles):
            texts = [article.get('title', '') + ' ' + article.get('description', '') for article in news_articles]
        else:
            texts = news_articles
            
        # Analyze sentiment
        sentiments = self.analyze_batch(texts)
        
        # Add sentiment to articles
        results = []
        for i, sentiment in enumerate(sentiments):
            if isinstance(news_articles, list) and all(isinstance(item, dict) for item in news_articles):
                result = news_articles[i].copy()
                result['sentiment'] = sentiment
            else:
                result = {'text': news_articles[i], 'sentiment': sentiment}
            results.append(result)
            
        return results
            
    def analyze_earnings_report(self, report_text):
        """
        Analyze sentiment in an earnings report.
        
        Args:
            report_text (str): Text of the earnings report.
            
        Returns:
            dict: Overall sentiment and section sentiments.
        """
        # Split report into sections (simplified approach)
        sections = self._split_into_sections(report_text)
        
        # Analyze sentiment for each section
        section_sentiments = {}
        for section_name, section_text in sections.items():
            section_sentiments[section_name] = self.analyze_text(section_text)
            
        # Analyze overall sentiment
        overall_sentiment = self.analyze_text(report_text[:5000])  # Analyze first 5000 chars for overall sentiment
        
        return {
            "overall": overall_sentiment,
            "sections": section_sentiments
        }
        
    def analyze_sec_filing(self, filing_text):
        """
        Analyze sentiment in an SEC filing.
        
        Args:
            filing_text (str): Text of the SEC filing.
            
        Returns:
            dict: Overall sentiment and section sentiments.
        """
        # For SEC filings, use the same approach as earnings reports
        return self.analyze_earnings_report(filing_text)
    
    def _split_into_sections(self, text):
        """
        Split a financial document into sections.
        
        Args:
            text (str): Document text.
            
        Returns:
            dict: Sections of the document.
        """
        # This is a simplified approach - in a real implementation,
        # you would use more sophisticated section detection
        
        # Look for common section headers
        section_headers = [
            "Summary", "Overview", "Financial Results", "Revenue", "Earnings",
            "Outlook", "Guidance", "Forward-Looking", "Risk Factors"
        ]
        
        sections = {}
        remaining_text = text
        
        for header in section_headers:
            # Find section in text
            start_idx = remaining_text.find(header)
            if start_idx != -1:
                # Find the end of the section (next section header or end of text)
                end_idx = len(remaining_text)
                for next_header in section_headers:
                    if next_header != header:
                        next_idx = remaining_text.find(next_header, start_idx + len(header))
                        if next_idx != -1 and next_idx < end_idx:
                            end_idx = next_idx
                
                # Extract section text
                section_text = remaining_text[start_idx:end_idx].strip()
                sections[header] = section_text
        
        # If no sections were found, use the whole text
        if not sections:
            sections["Full Document"] = text
            
        return sections
        
    def fine_tune(self, texts, labels, validation_texts=None, validation_labels=None, epochs=3, learning_rate=2e-5):
        """
        Fine-tune the FinBERT model on Indian market data.
        
        Args:
            texts (list): List of training texts.
            labels (list): List of training labels (0=bearish, 1=neutral, 2=bullish).
            validation_texts (list, optional): Validation texts.
            validation_labels (list, optional): Validation labels.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for fine-tuning.
            
        Returns:
            dict: Training metrics.
        """
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
        import evaluate
        
        # Convert labels to their numeric indices
        label_to_id = {label: i for i, label in enumerate(self.labels)}
        if isinstance(labels[0], str):
            numeric_labels = [label_to_id.get(label.lower(), 1) for label in labels]  # Default to neutral
        else:
            numeric_labels = labels
            
        # Create dataset
        train_dataset = Dataset.from_dict({
            "text": texts,
            "label": numeric_labels
        })
        
        # Create validation dataset if provided
        if validation_texts and validation_labels:
            if isinstance(validation_labels[0], str):
                numeric_val_labels = [label_to_id.get(label.lower(), 1) for label in validation_labels]
            else:
                numeric_val_labels = validation_labels
                
            val_dataset = Dataset.from_dict({
                "text": validation_texts,
                "label": numeric_val_labels
            })
        else:
            # Split training data for validation
            train_val_dataset = train_dataset.train_test_split(test_size=0.2)
            train_dataset = train_val_dataset["train"]
            val_dataset = train_val_dataset["test"]
            
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir="./results/finbert_finetuned",
            learning_rate=learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Setup metrics
        metric = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        train_results = trainer.train()
        
        # Evaluate model
        eval_results = trainer.evaluate()
        
        # Update model to fine-tuned version
        self.model = trainer.model
        
        # Return metrics
        return {
            "train_metrics": train_results.metrics,
            "eval_metrics": eval_results
        }

# Example usage
if __name__ == "__main__":
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Analyze single text
    text = "The company reported strong earnings, exceeding analyst expectations by 15%. Revenue growth was impressive at 25% year-over-year."
    sentiment = sentiment_analyzer.analyze_text(text)
    print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.4f})")
    
    # Analyze batch of texts
    texts = [
        "The company missed earnings estimates and lowered guidance for next quarter.",
        "The market remained stable today with mixed results across sectors.",
        "The tech sector surged on positive earnings reports from major players."
    ]
    sentiments = sentiment_analyzer.analyze_batch(texts)
    for i, sentiment in enumerate(sentiments):
        print(f"Text {i+1} Sentiment: {sentiment['label']} (Score: {sentiment['score']:.4f})") 