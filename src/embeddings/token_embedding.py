from torch import nn

class TokenEmbedding(nn.Embedding):
    """ class for token embeddings

    Args:
    - d_vocabulary (int): dimension of vocabulary space
    - d_model (int): dimension of embedding vector space
    """

    def __init__(self, d_vocabulary, d_model):
        
        super(TokenEmbedding, self).__init__(d_vocabulary, d_model, padding_idx = 1)