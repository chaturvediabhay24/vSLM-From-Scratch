import pandas as pd


class PreProcessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.corpus = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self

    def build_corpus(self):
        """Concatenate all stories into a single string separated by <EOS> tokens."""
        stories = self.data["text"].dropna().astype(str).tolist()
        self.corpus = "<EOS>".join(stories) + "<EOS>"
        return self

    def sample_corpus(self, n=100_000, random_state=42):
        """Build a corpus from a random sample of n stories (for tokenizer training)."""
        sample = self.data["text"].dropna().astype(str).sample(n=n, random_state=random_state)
        self.corpus = "<EOS>".join(sample.tolist()) + "<EOS>"
        return self

    def preprocess(self):
        self.load_data()
        self.build_corpus()
        return self.corpus
