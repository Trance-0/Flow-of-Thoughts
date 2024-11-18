from abc import ABC, abstractmethod
from language_models.abstract_language_model import AbstractLanguageModel
from thoughts.thought import Thought

class Challenge(ABC):
    def __init__(self, prompt: str, language_model: AbstractLanguageModel, goal: str):
        self.prompt = prompt
        self.language_model = language_model
        self.goal = goal
        self.desire = Thought(prompt, language_model)

    def __str__(self) -> str:
        return f"Challenge(prompt={self.prompt}, goal={self.goal})"