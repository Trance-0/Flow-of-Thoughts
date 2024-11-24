from abc import ABC, abstractmethod
from enum import Enum
import itertools
from typing import Callable, Dict, Iterator, List, Optional
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
    
    def __init__(self, parent_thought: Thought, language_model: AbstractLanguageModel, child_thought: List[Thought]=None):
        """
        Generate child node by defined operation from parent thought
        This is a one-to-one operation, so each operation has only one parent and one child.
        """
        self.language_model = language_model
        self.parent_thought = parent_thought
        self.child_thought = child_thought

    @abstractmethod
    def generate_child(self) -> float:
        """
        Generate child node by defined operation from parent thought, return cost of the operation
        """
        pass

    def __str__(self) -> str:
        return f"Operation(parent_thought={self.parent_thought}, child_thought={self.child_thought})"
    
    @abstractmethod
    def __json__(self) -> dict:
        pass
    
class Parser:
    """
    Abstract class for parsing the input and output of a language model.
    """
    def __init__(self, prompt: str, cot: str="", shots: List[dict]=[]):
        self.prompt = prompt
        self.cot = cot
        self.shots = shots

    def parse(self,input: str) -> str:
        res=f'<Instruction> {self.prompt} </Instruction>'
        if len(self.cot) > 0:
            res += f'\n<Approach>\n{self.cot}\n</Approach>'
        if len(self.shots) > 0:
            partial_examples=[]
            for shot in self.shots:
                current_example=""
                current_example+=f"Input: {shot['input']}\n"
                for key, value in shot.items():
                    if key != "input" and key != "output":
                        current_example+=f"{key}: {value}\n"
                current_example+=f"Output: {shot['output']}"
                partial_examples.append(current_example)
            res += '\n<Examples>\n'+\
            '\n'.join(partial_examples)+\
            '\n</Examples>'
        return res+f"\nInput: {input}"

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

    def generate_child(self) -> None:
        if self.scoring_function is not None:
            score = self.scoring_function(self.parent_thought)
        else:
            score = self.language_model.query(self.parent_thought, self.scoring_prompt)
        self.child_thought[0].content = score

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "scoring_function": self.scoring_function,
            "scoring_prompt": self.scoring_prompt,
            "child_thought": self.child_thought.__json__(),
        }

class ValidateAndImprove(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(self, parent_thought: Thought, language_model: AbstractLanguageModel, validate_function: Callable[[Dict], bool]=None, improve: bool=True, num_tries: int=3):
        """
        Initialize a new ValidateAndImprove operation.
        
        :param validate_function: A function to validate thoughts (if not using LM).
        :type validate_function: Callable[[Dict], bool]
        :param improve: Whether to improve the thought if it is not valid. Defaults to True.
        :type improve: bool
        :param num_tries: Number of tries to improve the thought before giving up. Defaults to 3.
        :type num_tries: int
        """
        self.validate_function = validate_function
        self.improve = improve
        self.num_tries = num_tries
        super().__init__(parent_thought, language_model)

    def generate_child(self) -> None:

        # TODO: continue from here
        parents=self.parent_thought
        for parent in parents:
            current_thought=Thought.from_thought(parent)
            current_try=0
            while True:
                if self.validate_function is not None:
                    valid=self.validate_function(current_thought.state)