import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/rwanda_fare_data.csv")

# Check if column exists
if "text" not in data.columns:
    raise Exception("Dataset must contain a 'text' column.")

# Clean and analyze sentiment
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

data['Sentiment'] = data['text'].apply(get_sentiment)

# Show sentiment counts
print(data['Sentiment'].value_counts())

# Save results
data.to_csv("data/sentiment_results.csv", index=False)

# Plot results
data['Sentiment'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/sentiment_plot.png")
plt.show()
