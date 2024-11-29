from abc import ABC, abstractmethod
from enum import IntEnum
import itertools
import random
from typing import Callable, Dict, Iterator, List, Optional, Any
from thoughts.thought import Thought
from language_models.abstract_language_model import AbstractLanguageModel
import logging

class OperationType(IntEnum):
    """
    Enum to represent different operation types that can be used as unique identifiers.
    """

    generate: int = 0
    evaluate: int = 1
    score: int = 2
    improve: int = 3
    validate: int = 4
    aggregate: int = 5
    keep_best_n: int = 6
    keep_valid: int = 7
    selector: int = 8

    def __str__(self) -> str:
        return self.name

class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None
    
    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, children_thoughts: Optional[List[Thought]]=None,input_size: int=-1,output_size: int=-1):
        """
        Generate child node by defined operation from parent thought
        This is a one-to-one operation, so each operation has only one parent and one child.

        :param parents_thoughts: The parent thoughts
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for generation
        :type language_model: AbstractLanguageModel
        :param children_thoughts: The child thoughts
        :type children_thoughts: List[Thought]
        :param input_size: The number of parent thoughts. must be greater than 0
        :type input_size: int
        :param output_size: The number of child thoughts. -1 means the same as input size
        :type output_size: int
        """
        self.language_model = language_model
        self.parents_thoughts = parents_thoughts
        self.children_thoughts = children_thoughts
        self.input_size = input_size
        self.output_size = output_size
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        # validate input size and output size
        if input_size < 1:
            raise ValueError("The input size must be greater than 0")
        if output_size < 1:
            output_size = input_size
        self.hash = random.randint(0, 2**64-1)
        self.connect_thoughts()
        self.validate_input()

    def validate_input(self) -> None:
        """
        Validate the number of parent thoughts and child thoughts in this method
        """
        if len(self.parents_thoughts) != self.input_size:
            raise ValueError(f"The number of parent thoughts must be {self.input_size}")
        if len(self.children_thoughts) != self.output_size:
            raise ValueError(f"The number of child thoughts must be {self.output_size}")

    def connect_thoughts(self) -> None:
        """
        Connect parent thoughts to the operation and attach the operation to the parent thoughts
        Validate the number of parent thoughts and child thoughts in this method
        """
        # auto generate children thoughts
        if len(self.children_thoughts)==0:
            if self.input_size == 1:
                self.logger.warning("No child thoughts provided,a new thought will be created")
                self.children_thoughts = [Thought("",[self],[])]
            else:
                self.logger.warning(f"No child thoughts provided, {self.output_size} new thoughts will be created")
                self.children_thoughts = [Thought("",[self],[]) for _ in range(self.output_size)]
        if self.input_size == 1:
            # connect thoughts with operation
            self.logger.info(f"Connecting {self.input_size} parent thoughts to {self}")
            for parent in self.parents_thoughts:
                parent.append_child_operation(self)

    @abstractmethod
    def generate_children(self) -> float:
        """
        Generate child node by defined operation from parent thought, 
        And assign value to self.child_thoughts if applicable

        :return: cost of the operation
        :rtype: float
        """
        self.logger.info(f"Generating child for operation: {self.operation_type}")
        pass

    def is_executable(self) -> bool:
        return all([parent.is_executable for parent in self.parents_thoughts])

    def get_children_thoughts(self) -> List[Thought]:
        return self.children_thoughts
    
    def get_parents_thoughts(self) -> List[Thought]:
        return self.parents_thoughts
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def __json__(self) -> dict:
        pass
    
class Parser:
    """
    Abstract class for parsing the input and output of a language model.
    """
    def __init__(self, prompt: str, cot: str="", shots: List[dict]=[],name: str="unnamed"):
        self.prompt = prompt
        self.cot = cot
        self.shots = shots
        self.name = name
        self.logger = logging.getLogger(f'{self.__class__.__name__}')

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
        res=res+f"\nInput: {input}"
        self.logger.info(f"Parser input: {input}, final prompt: {res}")
        return res

    @staticmethod
    def from_dict(prompt: dict) -> 'Parser':
        return Parser(prompt["instruction"], prompt.get("approach", ""), prompt.get("examples", []), name=prompt.get("name", "unnamed"))

    def __str__(self) -> str:
        return f"Parser(prompt={self.prompt}, cot={self.cot}, shots={self.shots}, name={self.name})"


class Generate(Operation):
    """
    Operation to generate thoughts.

    Input: one thought to generate children from 
    Output: children thoughts
    """

    operation_type: OperationType = OperationType.generate

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, branching_factor: int, generate_prompt: Optional['Parser']=None, children_thoughts: List[Thought]=[]):
        """
        Initialize a new Generate operation.

        :param parents_thoughts: The parent thought to generate children from, in our implementation, the length of the parent thoughts must be one.
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for generation.
        :type language_model: AbstractLanguageModel
        :param branching_factor: The branching factor for generation, the number of children thoughts to generate for each parent thought.
        :type branching_factor: int
        :param generate_prompt_name: The name of the generate prompt when generating json.
        :type generate_prompt_name: str
        :param children_thoughts: The child thoughts to generate.
        :type children_thoughts: List[Thought]
        """
        self.branching_factor = branching_factor
        self.generate_prompt = generate_prompt
        # default name for unnamed generate prompt
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=1, output_size=branching_factor)

    def generate_children(self) -> float:
        self.logger.info(f"Generating {self.branching_factor} children for {self.parents_thoughts}")
        cost = 0
        for i in range(self.branching_factor):
            child = self.children_thoughts[i]
            query = self.language_model.query(self.generate_prompt.parse(self.parents_thoughts[0].content))
            child.content = self.language_model.get_response_texts(query)
            child.is_executable = True
            cost += self.language_model.get_response_cost(query)
        return cost
    
    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "generate_prompt_name": self.generate_prompt.name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }
    
    def __str__(self) -> str:
        return f"Generate(generate_prompt_name={self.generate_prompt.name}, children_thoughts=[{len(self.children_thoughts)}]={','.join([str(child) for child in self.children_thoughts])})"
    
class Evaluate(Operation):
    """
    Operation to evaluate thoughts by comparing the output of the thought with the truth.
    """

    operation_type: OperationType = OperationType.evaluate

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, evaluate_prompt: Optional['Parser']=None, evaluate_prompt_name: str="unnamed", evaluate_function: Callable[[Thought,Any], float]=None, ground_truth: Any=None, children_thoughts: List[Thought]=[]):
        """
        Initialize a new Evaluate operation.

        :param parents_thoughts: The parent thoughts to evaluate in this evaluation operation, the length of the parent thoughts must be one.
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for evaluation.
        :type language_model: AbstractLanguageModel
        :param evaluate_prompt: A parser to evaluate thoughts.
        :type evaluate_prompt: Parser
        :param evaluate_prompt_name: The name of the evaluate prompt when generating json.
        :type evaluate_prompt_name: str
        :param evaluate_function: A function to evaluate thoughts.
        :type evaluate_function: Callable[[Thought], float]
        :param ground_truth: The ground truth to evaluate thoughts.
        :type ground_truth: Any
        :param children_thoughts: The child thoughts to evaluate.
        :type children_thoughts: List[Thought]
        """
        if (evaluate_function is None) and (evaluate_prompt is None):
            raise ValueError("Either evaluate_function or evaluate_prompt must be provided")
        if (evaluate_function is not None) and (evaluate_prompt is not None):
            raise ValueError("Only one of evaluate_function or evaluate_prompt can be provided")
        if evaluate_function is not None:
            self.evaluate_function = evaluate_function
            if ground_truth is None:
                raise ValueError("ground_truth must be provided when evaluate_function is used")
            self.ground_truth = ground_truth
        if evaluate_prompt is not None:
            self.evaluate_prompt = evaluate_prompt
            self.evaluate_prompt_name = evaluate_prompt_name
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=1, output_size=1)

    def generate_children(self) -> float:
        self.logger.info(f"Evaluating thought: {self.parents_thoughts}")
        cost = 0
        score = None
        if self.evaluate_function is not None:
            score = self.evaluate_function(self.parents_thoughts[0], self.ground_truth)
        else:
            query = self.language_model.query(self.evaluate_prompt.parse(self.parents_thoughts[0].content))
            score = self.language_model.get_response_texts(query)
            cost += self.language_model.get_response_cost(query)
        self.children_thoughts[0].content = score
        self.children_thoughts[0].is_executable = True
        return cost

    def __json__(self) -> dict:
        if self.evaluate_function is not None:  
            return {
                "operation_type": self.operation_type,
                "evaluate_function": self.evaluate_function.__name__,
                "ground_truth": self.ground_truth,
                "children_thoughts": [child.__json__() for child in self.children_thoughts],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "evaluate_prompt_name": self.evaluate_prompt.name,
                "children_thoughts": [child.__json__() for child in self.children_thoughts],
            }
    
    def __str__(self) -> str:
        if self.evaluate_function is not None:
            return f"Evaluate(evaluate_function={self.evaluate_function}, ground_truth={self.ground_truth}, children_thoughts={self.children_thoughts})"
        else:
            return f"Evaluate(evaluate_prompt_name={self.evaluate_prompt.name}, children_thoughts={self.children_thoughts})"    

class Score(Operation):
    """
    Operation to score thoughts.

    Input: thoughts
    Output: thoughts with scores
    """

    operation_type: OperationType = OperationType.score

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, scoring_function: Callable[[Thought], float]=None, scoring_prompt: Optional['Parser']=None, children_thoughts: List[Thought]=[]):
        """
        Initialize a new Score operation.

        :param parents_thoughts: The parent thoughts to score length must be 1, will only generate one child thought.
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for scoring.
        :type language_model: AbstractLanguageModel
        :param is_one_to_one: Whether the operation is one-to-one, that is, whether the length of the parent thoughts and child thoughts must be the same. Defaults to True.
        :type is_one_to_one: bool
        :param scoring_function: A function to score thoughts.
        :type scoring_function: Callable[[Thought], float]
        :param scoring_prompt: A parser to score thoughts.
        :type scoring_prompt: Parser
        :param children_thoughts: The child thoughts to score length must be 1, will only generate one child thought.
        :type children_thoughts: List[Thought]
        """
        self.scoring_function = scoring_function
        self.scoring_prompt = scoring_prompt
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=1, output_size=1)


    def generate_children(self) -> float:
        self.logger.info(f"Scoring thought: {self.parents_thoughts}")
        cost = 0
        if self.scoring_function is not None:
            score = self.scoring_function(self.parents_thoughts[0])
        else:
            query = self.language_model.query(self.scoring_prompt.parse(self.parents_thoughts[0].content))
            score = self.language_model.get_response_texts(query)
            cost += self.language_model.get_response_cost(query)
        self.children_thoughts[0].content = score
        # score thoughts is not executable
        self.children_thoughts[0].is_executable = False
        return cost

    def __json__(self) -> dict:
        if self.scoring_function is not None:
            return {
                "operation_type": self.operation_type,
                "scoring_function": self.scoring_function,
                "children_thoughts": [child.__json__() for child in self.children_thoughts],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "scoring_prompt_name": self.scoring_prompt_name,
                "children_thoughts": [child.__json__() for child in self.children_thoughts],
            }
    
    def __str__(self) -> str:
        if self.scoring_function is not None:
            return f"Score(scoring_function={self.scoring_function}, children_thoughts={self.children_thoughts})"
        else:
            return f"Score(scoring_prompt_name={self.scoring_prompt.name}, children_thoughts={self.children_thoughts})"
    
class Improve(Operation):
    """
    Operation to improve thoughts by querying a language model. (assume the refined thought is better)
    Comparing with ValidateAndImprove, this operation only use the improve_prompt once for each parent thought.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, improve_prompt: Optional['Parser']=None, children_thoughts: List[Thought]=[]):
        """
        Initialize a new Improve operation.

        :param parents_thoughts: The parent thoughts to improve.
        :type parents_thoughts: List[Thought]
        :param improve_prompt: A parser to improve thoughts.
        :type improve_prompt: Parser
        :param improve_prompt_name: The name of the improve prompt when generating json.
        :type improve_prompt_name: str
        :param children_thoughts: The child thoughts to improve, must have equal length as parents_thoughts.
        :type children_thoughts: List[Thought]
        """
        if improve_prompt is None:
            raise ValueError("improve_prompt must be provided")
        self.improve_prompt = improve_prompt
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=1, output_size=1)

    def generate_children(self) -> float:
        self.logger.info(f"Improving thought: {self.parents_thoughts}")
        cost = 0    
        query = self.language_model.query(self.improve_prompt.parse(self.parents_thoughts[0].content))
        self.children_thoughts[0].content = self.language_model.get_response_texts(query)
        cost += self.language_model.get_response_cost(query)
        return cost
    
    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "improve_prompt_name": self.improve_prompt.name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def __str__(self) -> str:
        return f"Improve(improve_prompt_name={self.improve_prompt.name}, children_thoughts={self.children_thoughts})"

class Validate(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, validate_function: Callable[[str,str], bool]=None, validate_prompt: Optional['Parser']=None, children_thoughts: List[Thought]=[]):
        """
        Initialize a new Validate operation.
        
        :param parents_thoughts: The parent thoughts to validate and improve.
        :type parents_thoughts: List[Thought]
        :param validate_function: A function to validate thoughts (if not using LM).
        :type improve_operation: Operation
        :param validate_function: A function to validate thoughts (if not using LM).
        :type validate_function: Callable[[Dict], bool]
        :param validate_prompt: A parser to validate thoughts.
        :type validate_prompt: Parser
        :param children_thoughts: The child thoughts to validate and improve, must have equal length as parents_thoughts.
        :type children_thoughts: List[Thought]
        """
        if validate_function is None and validate_prompt is None:
            raise ValueError("validate_function or validate_prompt must be provided")
        if validate_function is not None and validate_prompt is not None:
            raise ValueError("validate_function and validate_prompt cannot be provided at the same time")
        self.validate_function = validate_function
        self.validate_prompt = validate_prompt
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=1, output_size=1)


    def generate_children(self) -> float:
        parent=self.parents_thoughts[0]
        cost = 0
        if self.validate_function is not None:
            self.children_thoughts[0].content = self.validate_function(parent.content)
        else:
            query = self.language_model.query(self.validate_prompt.parse(parent.content))
            cost += self.language_model.get_response_cost(query)
            self.children_thoughts[0].content = self.language_model.get_response_texts(query)
        self.children_thoughts[0].is_executable = False
        self.logger.info(f"Validated thought: {self.children_thoughts}")
        return cost

    def __str__(self) -> str:
        if self.validate_function is not None:
            return f"Validate(validate_function={self.validate_function.__name__})"
        else:
            return f"Validate(validate_prompt_name={self.validate_prompt.name})"
    
    def __json__(self) -> dict:
        if self.validate_function is not None:
            return {
                "operation_type": self.operation_type,
                "validate_function": self.validate_function,
            }
        else:
            return {
                "operation_type": self.operation_type,
                "validate_prompt_name": self.validate_prompt.name,
            }

class Aggregate(Operation):
    """
    Operation to aggregate thoughts.
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, aggregate_prompt: Optional['Parser'], children_thoughts: List[Thought]=[]):
        """
        Initialize a new Aggregate operation.
        """
        if aggregate_prompt is None:
            raise ValueError("aggregate_prompt must be provided")
        self.aggregate_prompt = aggregate_prompt
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=2, output_size=1)

    def generate_children(self) -> float:
        self.logger.info(f"Aggregating thoughts: {self.parents_thoughts}")
        query = self.language_model.query(self.aggregate_prompt.parse("\n".join([parent.content for parent in self.parents_thoughts])))
        self.children_thoughts[0].content = self.language_model.get_response_texts(query)
        return self.language_model.get_response_cost(query)
    
    def __str__(self) -> str:
        return f"Aggregate(aggregate_prompt_name={self.aggregate_prompt.name}, children_thoughts={self.children_thoughts})"
    
    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "aggregate_prompt_name": self.aggregate_prompt.name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }
    

class KeepBestN(Operation):
    """
    Operation to keep the best N thoughts. Must score the thoughts before proceeding this operation.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, n: int, highest_score_first: bool=True, children_thoughts: List[Thought]=[]):
        """
        Initialize a new KeepBestN operation.
        
        :param parents_thoughts: The parent thoughts to keep the best N.
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for keeping the best N thoughts.
        :type language_model: AbstractLanguageModel
        :param n: The number of thoughts to keep.
        :type n: int
        :param highest_score_first: Whether to keep the thoughts with the highest score first.
        :type highest_score_first: bool
        """
        for parent in parents_thoughts:
            if len(parent.get_parents_operations())!=1 or parent.get_parents_operations()[0].operation_type!=OperationType.score:
                raise ValueError("All parent thoughts must be scored")
        self.n = n
        self.highest_score_first = highest_score_first
        super().__init__(parents_thoughts, language_model, children_thoughts, input_size=len(parents_thoughts), output_size=n)

    def generate_children(self) -> float:
        self.logger.info(f"Keeping best {self.n} thoughts: {self.parents_thoughts}")
        q=[]
        for parent in self.parents_thoughts:
            q.append((float(parent.content), parent))
        q.sort(key=lambda x: x[0], reverse=self.highest_score_first)
        for i in range(self.n):
            # going up two levels, ignoring the parent scoring layer
            self.children_thoughts[i].content = q[i][1].get_parents_thoughts()[0].get_parents_operations()[0].get_parents_thoughts()[0].content
            self.children_thoughts[i].is_executable = True
        return 0

class KeepValid(Operation):
    """
    Operation to keep valid thoughts.
    """

    operation_type: OperationType = OperationType.keep_valid

class Selector(Operation):
    """
    Operation to select thoughts.
    """

    operation_type: OperationType = OperationType.selector
