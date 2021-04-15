# %%
from transformers import ElectraModel, ElectraTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.vocab_size = 35000
        
    def get_token_ids(self, string):
        tokens = self.tokenizer.tokenize(string)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

# %%
