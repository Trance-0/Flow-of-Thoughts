from abc import ABC, abstractmethod
from enum import Enum
import itertools
from typing import Callable, Iterator, List, Optional
from thoughts.thought import Thought
from language_models.abstract_language_model import AbstractLanguageModel

class OperationType(Enum):
    """
    Enum to represent different operation types that can be used as unique identifiers.

    This 
    """

    score: int = 0
    validate_and_improve: int = 1
    generate: int = 2
    improve: int = 3
    aggregate: int = 4
    keep_best_n: int = 5
    keep_valid: int = 6
    ground_truth_evaluator: int = 7
    selector: int = 8

class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None
    
    def __init__(self, parent_thought: Thought, language_model: AbstractLanguageModel):
        """
        Generate child node by defined operation from parent thought
        This is a one-to-one operation, so each operation has only one parent and one child.
        """
        self.language_model = language_model
        self.parent_thought = parent_thought
        self.child_thought = self.__generate_child()

    @abstractmethod
    def __generate_child(self) -> Thought:
        pass

    def __str__(self) -> str:
        return f"Operation(parent_thought={self.parent_thought}, child_thought={self.child_thought})"
    
class Parser:
    """
    Abstract class for parsing the input and output of a language model.
    """
    def __init__(self, prompt: str, shots: List[tuple[str, str]]):
        self.prompt = prompt
        self.shots = shots

    def parse(self) -> str:
        return f'<Instruction> {self.prompt} </Instruction>\n\n<Examples>\n'+\
            '\n'.join([f"Input: {shot[0]}\nOutput: {shot[1]}" for shot in self.shots])+\
            '\n</Examples>'

    def __str__(self) -> str:
        return self.parse()

class Score(Operation):
    """
    Operation to score thoughts.

    Input: One thought
    Output: One score for the thought
    """

    operation_type: OperationType = OperationType.score

    def __init__(self, parent_thought: Thought, language_model: AbstractLanguageModel, scoring_function: Callable[[Thought], float]=None, scoring_prompt: Optional['Parser']=None):
        if (scoring_function is None) and (scoring_prompt is None):
            raise ValueError("Either scoring_function or scoring_prompt must be provided")
        if (scoring_function is not None) and (scoring_prompt is not None):
            raise ValueError("Only one of scoring_function or scoring_prompt can be provided")
        self.scoring_function = scoring_function
        self.scoring_prompt = scoring_prompt
        super().__init__(parent_thought, language_model)

    def __generate_child(self) -> Thought:
        if self.scoring_function is not None:
            score = self.scoring_function(self.parent_thought)
        else:
            score = self.language_model.query(self.parent_thought, self.scoring_prompt)
        return Thought(self.parent_thought.content, self.language_model, score)
