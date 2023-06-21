import torch.nn as nn

class Embedder():
    def __init__(self, embedder):
        self.embedder = embedder
    
    def embed(self, input_ids, additional_info):
        word_embeddings = self.embedder.word_embeddings(input_ids)
        #add additional info
        return word_embeddings
    
class Classifier():
    def __init__(self, model, classifier):
        self.model = model
        self.classifier = classifier

    def classify(self, embeddings, attention_mask):
        encoding = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
        last_hidden_state = encoding.last_hidden_state
        logits = self.classifier(last_hidden_state)
        return logits

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.embedder = Embedder(model.roberta.embeddings)
        self.classifier = Classifier(model.roberta, model.classifier)

    def forward(self, input_ids, attention_mask, additional_info=None):
        embeddings = self.embedder.embed(input_ids, additional_info)
        logits = self.classifier.classify(embeddings, attention_mask)
        return logits
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
