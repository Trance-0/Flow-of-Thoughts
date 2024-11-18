from test.check_models import check_models
from language_models.chatgpt import ChatGPT
from language_models.llamachat_hf import LlamaHF

if __name__ == "__main__":
    # check_models(ChatGPT(model_name="chatgpt4o"))
    check_models(LlamaHF(model_name="llama3.2-1b-hf"))
