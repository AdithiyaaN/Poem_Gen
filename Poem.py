import random
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample corpus of text
corpus = """
Whose woods these are I think I know.
His house is in the village though;
He will not see me stopping here
To watch his woods fill up with snow.

My little horse must think it queer
To stop without a farmhouse near
Between the woods and frozen lake
The darkest evening of the year.

He gives his harness bells a shake
To ask if there is some mistake.
The only other sound's the sweep
Of easy wind and downy flake.

The woods are lovely, dark and deep,
But I have promises to keep,
And miles to go before I sleep,
And miles to go before I sleep.
"""

def normalize_text(corpus):
    """
    Normalize the text by removing punctuation and converting to lowercase.
    """
    corpus = re.sub(r'[\W_]+', ' ', corpus)
    return corpus.lower()

# Normalize and tokenize the corpus
tokens = word_tokenize(normalize_text(corpus))

# Create frequency-weighted bigram Markov chain model
weighted_bigram_model = defaultdict(Counter)
for i in range(len(tokens) - 2):
    bigram = (tokens[i], tokens[i + 1])
    next_word = tokens[i + 2]
    weighted_bigram_model[bigram][next_word] += 1

def weighted_random_choice(counter):
    """
    Choose a random word from the counter based on weights.
    """
    total = sum(counter.values())
    rand = random.uniform(0, total)
    upto = 0
    for choice, weight in counter.items():
        if upto + weight >= rand:
            return choice
        upto += weight

def generate_weighted_text(weighted_bigram_model, start_bigram, num_words=50):
    """
    Generate text using the weighted bigram Markov chain model.
    """
    current_bigram = start_bigram
    output = list(current_bigram)

    for _ in range(num_words - 2):
        if current_bigram in weighted_bigram_model:
            next_word = weighted_random_choice(weighted_bigram_model[current_bigram])
            output.append(next_word)
            current_bigram = (current_bigram[1], next_word)
        else:
            break

    return ' '.join(output)

def format_poem(text, line_length=7):
    """
    Format the generated text into lines of specified length.
    """
    words = text.split()
    poem = []
    for i in range(0, len(words), line_length):
        line = ' '.join(words[i:i + line_length])
        poem.append(line)
    return '\n'.join(poem)

# Generate text and format it as a poem
start_bigram = ('woods', 'fill')
generated_weighted_text = generate_weighted_text(weighted_bigram_model, start_bigram)
formatted_poem = format_poem(generated_weighted_text)
print(formatted_poem)
