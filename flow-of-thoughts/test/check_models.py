from language_models.chatgpt import ChatGPT
from language_models.llamachat_hf import LlamaHF
from language_models.abstract_language_model import AbstractLanguageModel

def check_models(model: AbstractLanguageModel):
    print(model.query("What is the meaning of life?"))

