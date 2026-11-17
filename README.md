# ChatBot: coffee and tea delivery

A small intent-classification chatbot for a coffee and tea delivery service. You type a question, it decides which intent you meant, and it replies from a set of canned answers covering greetings, item availability, payment methods, delivery windows, and complaints.

It is a classic bag-of-words classifier, not a language model. That is the point: it shows the full loop of building an intent classifier from scratch, with a dataset you can read.

## How it works

```text
sentence
   │  NLTK word_tokenize
   ▼
Porter stemmer  ──►  bag-of-words over the training vocabulary  ──►  PyTorch MLP  ──►  intent
```

| File | What it does |
|---|---|
| `intents.json` | The training data: intents, example patterns, and the responses each intent can return |
| `nltk_utils.py` | Tokenisation, Porter stemming and the bag-of-words vectoriser |
| `model.py` | A three-layer feedforward network (ReLU, no softmax on the output, because the loss takes raw logits) |
| `train.py` | Trains on `intents.json` and saves the weights plus the vocabulary to `data.pth` |
| `chat.py` | Loads `data.pth` and holds the conversation |

## Running it

```bash
pip install torch nltk numpy
python train.py     # writes data.pth
python chat.py      # talk to it
```

`train.py` has to run first, because `chat.py` loads `data.pth` rather than training on start.

## Limitations, honestly

- It has no context. Each message is classified on its own, so "and how long does delivery take?" only works if the words themselves carry enough signal.
- The vocabulary is fixed at training time. A word the model has never seen contributes nothing.
- Adding an intent means editing `intents.json` and retraining, which is fine at this scale and would not be at ten thousand intents.

For a version where the language model does the understanding, see [MA-CDH](https://github.com/Arjun-Sivasankar/MA-CDH).
