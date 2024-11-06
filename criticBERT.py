from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Load the fine-tuned tokenizer and model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 2. Function to predict the sentiment for a batch of reviews
def predict_batch_sentiment(reviews):
    results = []
    for review in reviews:
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = torch.argmax(predictions).item()
        
        # Sentiment interpretation
        if sentiment_score >= 3:
            sentiment = "Positive"
        elif sentiment_score == 2:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        
        results.append((review, sentiment))
    return results

# 3. List of new reviews
more_reviews = [
    "What a disaster! The story made no sense and the acting was wooden.",
    "I really enjoyed this movie. The characters were relatable and the plot was engaging.",
    "It was a decent watch, but not something I would see again.",
    "Horrible. Just horrible. I want my time back.",
    "An exceptional movie that touched my heart. Highly recommended!",
    "The soundtrack was amazing, but the movie itself was just average.",
    "A predictable and dull experience. I've seen better stories in commercials.",
    "This movie is a must-watch! It has everything: drama, action, and a heartwarming message.",
    "Completely overrated. I don't understand the hype around this film.",
    "An underwhelming experience. It had potential but failed to deliver.",
    "Stunning visuals and a captivating story. I couldn't take my eyes off the screen.",
    "Terrible pacing and underdeveloped characters. Not worth your time.",
    "I loved every minute of it. A beautiful journey from start to finish.",
    "The humor fell flat, and the plot was all over the place.",
    "A great movie for the whole family. Everyone will find something to enjoy.",
    "One of the worst films I've ever seen. A complete waste of time.",
    "Simply outstanding! The direction and performances were top-notch.",
    "Boring and unimaginative. The film lacked any real substance.",
    "It had its moments, but overall it was forgettable.",
    "An absolute masterpiece. The best movie I've seen in years.",
    "The film started strong but lost its way in the second half.",
    "Poorly written and poorly acted. A real disappointment.",
    "Incredible! This movie exceeded all my expectations.",
    "Not bad, but not great either. A solid 6/10.",
    "This film redefines the genre. A revolutionary piece of cinema.",
    "Why did they even bother making this? It was awful.",
    "Charming, funny, and emotional. A true gem of a movie.",
    "The ending was unsatisfying, but the journey was enjoyable.",
    "The movie tried too hard to be clever, and it just came off as pretentious.",
    "An absolute borefest. I almost fell asleep halfway through."
]

# Analyze these reviews with the model
more_results = predict_batch_sentiment(more_reviews)

# Display the results
for review, sentiment in more_results:
    print(f"Review: {review}\nSentiment: {sentiment}\n")
