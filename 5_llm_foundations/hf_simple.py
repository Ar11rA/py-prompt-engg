from transformers import pipeline

# Initialize a pipeline for sentiment-analysis
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze the sentiment of a sentence
sentence = "I love using Hugging Face's Transformers library!"
result = sentiment_pipeline(sentence)

print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}")
