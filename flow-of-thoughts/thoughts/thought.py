from language_models.abstract_language_model import AbstractLanguageModel
from typing import Optional, List
from abc import ABC, abstractmethod
from thoughts.operations import Operation
class Thought(ABC):
    """
    A thought is a node in a flow of thoughts, the root thought is the initial prompt.
    """
    def __init__(self, content: str, parents: List[Optional['Operation']] = None, is_executable: bool=False):
        """
        Initialize a thought.
        
        :param content: The content of the thought.
        :type content: str
        :param language_model: The language model to use for the thought.
        :type language_model: AbstractLanguageModel
        :param parents: The parents of the thought. Root thoughts have no parents.
        :type parents: List[Optional['Operation']]
        :param is_executable: Whether the thought is executable, for example, thought after scoring is not executable because that's the final result. Initially, all thoughts are non-executable except for the root thought. It is executed after parent operations are executed.
        :type is_executable: bool
        """
        self.content = content
        self.parents = parents
        self.children = []
        self.is_executable = is_executable

    def get_children(self) -> List['Thought']:
        return [op.child_thought for op in self.children]
    
    def append_child(self, child: Operation) -> None:
        self.children.append(child)

    def append_children(self, children: List[Operation]) -> None:
        self.children.extend(children)
    
    def get_parents(self) -> List['Thought']:
        return [op.parent_thought for op in self.parents]

    @abstractmethod
    def compress(self) -> None:

        pass

    def __str__(self) -> str:
        return self.content
    
    def __json__(self) -> dict:
        return {
            "content": self.content,
            "language_model": self.language_model,
            "children": [child.__json__() for child in self.children],
        }
