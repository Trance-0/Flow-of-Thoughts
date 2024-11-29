import logging
import random
from language_models.abstract_language_model import AbstractLanguageModel
from typing import Optional, List

class Thought():
    """
    A thought is a node in a flow of thoughts, the root thought is the initial prompt.
    """
    def __init__(self, content: str, parents_operations:List, children_operations: List=None, is_executable: bool=False):
        """
        Initialize a thought.
        
        :param content: The content of the thought.
        :type content: str
        :param parents_operations: The parent operations of the thought. Root thoughts have no parent operations.
        :type parents_operations: List[Optional['Operation']]
        :param children_operations: The children operations of the thought.
        :type children_operations: List[Optional['Operation']]
        :param is_executable: Whether the thought is executable, for example, thought after scoring is not executable because that's the final result. Initially, all thoughts are non-executable except for the root thought. It is executed after parent operations are executed.
        :type is_executable: bool
        """
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.logger.debug(f"Parameters received: content={content}, parents_operations={parents_operations}, is_executable={is_executable}, children_operations={children_operations}")
        self.content = content
        self.parents_operations = parents_operations
        self.children_operations = children_operations
        self.is_executable = is_executable
        self.hash = random.randint(0, 2**64-1)
        self.logger.debug(f"Thought initialized with content: {self.content}, parents_operations: {','.join([str(pop) for pop in self.get_parents_operations()])}, is_executable: {self.is_executable}, children_operations: {','.join([str(cop) for cop in self.get_children_operations()])}")
    
    def get_children_operations(self) -> List:
        return self.children_operations if self.children_operations is not None else []
    
    def append_child_operation(self, child) -> None:
        self.logger.info(f"Trying to append child operation {child} to parent {self}")
        if child.hash not in [op.hash for op in self.get_children_operations()]:
            self.children_operations.append(child)
            self.logger.debug(f"Appended child operation {child} to parent {self}")
        else:
            self.logger.warning("Hash collision detected for {child}, child already exists. Request rejected.")

    def get_parents_operations(self) -> List:
        return self.parents_operations if self.parents_operations is not None else []
    
    def append_parent_operation(self, parent) -> None:
        if parent.hash not in [op.hash for op in self.get_parents_operations()]:
            self.parents_operations.append(parent)
        self.logger.debug(f"Appending parent operation {parent} to child {self}")

    def compress(self) -> None:
        pass

    def __str__(self) -> str:
        return f"Thought({self.content},children_operations=[{len(self.get_children_operations())}]=[{','.join([str(op) for op in self.get_children_operations()])}]"
    
    def __json__(self) -> dict:
        return {
            "content": self.content,
            "is_executable": self.is_executable,
            "children_operations": [op.__json__() for op in self.children_operations],
        }
