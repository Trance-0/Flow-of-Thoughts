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
    validate_and_improve: int = 4
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
    
    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, children_thoughts: Optional[List[Thought]]=None):
        """
        Generate child node by defined operation from parent thought
        This is a one-to-one operation, so each operation has only one parent and one child.
        """
        self.language_model = language_model
        self.parents_thoughts = parents_thoughts
        self.children_thoughts = children_thoughts
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.hash = random.randint(0, 2**64-1)
        self.connect_thoughts()

    @abstractmethod
    def connect_thoughts(self) -> None:
        """
        Connect parent thoughts to the operation and attach the operation to the parent thoughts
        Validate the number of parent thoughts and child thoughts in this method
        """
        pass

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
    def __init__(self, prompt: str, cot: str="", shots: List[dict]=[]):
        self.prompt = prompt
        self.cot = cot
        self.shots = shots
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
        self.logger.info(f"Parser input: {input}, final prompt: {res}")
        return res+f"\nInput: {input}"

    @staticmethod
    def from_dict(prompt: dict) -> 'Parser':
        return Parser(prompt["instruction"], prompt.get("approach", ""), prompt.get("examples", []))

    def __str__(self) -> str:
        return self.parse()


class Generate(Operation):
    """
    Operation to generate thoughts.

    Input: one thought to generate children from 
    Output: children thoughts
    """

    operation_type: OperationType = OperationType.generate

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, branching_factor: int, generate_prompt: Optional['Parser']=None, generate_prompt_name: str="unnamed", children_thoughts: List[Thought]=[]):
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
        if len(parents_thoughts)==0:
            raise ValueError("No parent thoughts provided")
        if len(parents_thoughts)>1:
            raise ValueError("The length of the parent thoughts must be one")
        self.branching_factor = branching_factor
        self.generate_prompt = generate_prompt
        # default name for unnamed generate prompt
        self.generate_prompt_name = generate_prompt_name
        super().__init__(parents_thoughts, language_model, children_thoughts)

    def connect_thoughts(self) -> None:
        # auto generate children thoughts
        if len(self.children_thoughts)==0:
            self.logger.warning(f"No child thoughts provided, {self.branching_factor} new thoughts will be created")
            # define children thoughts's parents operations to be self
            self.children_thoughts = [Thought("",parents_operations=[self]) for _ in range(self.branching_factor)]
        # connect thoughts with operation
        self.parents_thoughts[0].append_child_operation(self)
        self.logger.info(f"Thought connection done {self.parents_thoughts[0]}")

    def generate_children(self) -> float:
        self.logger.info(f"Generating {self.branching_factor*len(self.parents_thoughts)} children for {self.parents_thoughts}")
        cost = 0
        for i in range(self.branching_factor):
            child = self.children_thoughts[i]
            query = self.language_model.query(child, self.generate_prompt)
            child.content = self.language_model.get_response_texts(query)
            child.is_executable = True
            cost += self.language_model.get_response_cost(query)
        return cost
    
    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "generate_prompt_name": self.generate_prompt_name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }
    
    def __str__(self) -> str:
        return f"Generate(generate_prompt_name={self.generate_prompt_name}, children_thoughts=[{len(self.children_thoughts)}]={','.join([str(child) for child in self.children_thoughts])})"
    
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
        if len(parents_thoughts)==0:
            raise ValueError("No parent thoughts provided")
        if len(parents_thoughts)>1:
            raise ValueError("The length of the parent thoughts must be one")
        if len(children_thoughts)>1:
            raise ValueError("The length of the child thoughts must be one")
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
        super().__init__(parents_thoughts, language_model, children_thoughts)

    def connect_thoughts(self) -> None:
        if len(self.children_thoughts)==0:
            self.logger.warning(f"No child thoughts provided, {len(self.parents_thoughts)} new thoughts will be created")
            self.children_thoughts = [Thought("") for _ in range(len(self.parents_thoughts))]
        self.parents_thoughts[0].append_child_operation(self)
        self.children_thoughts[0].append_parent_operation(self)

    def generate_children(self) -> float:
        self.logger.info(f"Evaluating thought: {self.parents_thoughts}")
        cost = 0
        score = None
        if self.evaluate_function is not None:
            score = self.evaluate_function(self.parents_thoughts[0], self.ground_truth)
        else:
            query = self.language_model.query(self.parents_thoughts[0], self.evaluate_prompt)
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
                "evaluate_prompt_name": self.evaluate_prompt_name,
                "children_thoughts": [child.__json__() for child in self.children_thoughts],
            }
    
    def __str__(self) -> str:
        if self.evaluate_function is not None:
            return f"Evaluate(evaluate_function={self.evaluate_function}, ground_truth={self.ground_truth}, children_thoughts={self.children_thoughts})"
        else:
            return f"Evaluate(evaluate_prompt_name={self.evaluate_prompt_name}, children_thoughts={self.children_thoughts})"    

class Score(Operation):
    """
    Operation to score thoughts.

    Input: thoughts
    Output: thoughts with scores
    """

    operation_type: OperationType = OperationType.score

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, is_one_to_one: bool=True, scoring_function: Callable[[Thought], float]=None, scoring_prompt: Optional['Parser']=None, children_thoughts: List[Thought]=[]):
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
        if len(parents_thoughts)==0:
            raise ValueError("No parent thoughts provided")
        if (scoring_function is None) and (scoring_prompt is None):
            raise ValueError("Either scoring_function or scoring_prompt must be provided")
        if (scoring_function is not None) and (scoring_prompt is not None):
            raise ValueError("Only one of scoring_function or scoring_prompt can be provided")
        self.scoring_function = scoring_function
        self.scoring_prompt = scoring_prompt
        self.is_one_to_one = is_one_to_one
        super().__init__(parents_thoughts, language_model, children_thoughts)

    def connect_thoughts(self) -> None:
        # auto generate children thoughts
        if len(self.children_thoughts)==0:
            if self.is_one_to_one:
                n=len(self.parents_thoughts)    
                self.logger.warning(f"No child thoughts provided, {n} new thoughts will be created")
                self.children_thoughts = [Thought(None, [self]) for _ in range(n)]
            else:
                self.logger.warning("No child thoughts provided, a new thought will be created")
                self.children_thoughts = [Thought(None, [self])]
        # connect thoughts with operation
        if len(self.parents_thoughts)>1:
            if len(self.children_thoughts)==1:
                if self.is_one_to_one:
                    raise ValueError("The number of parent thoughts and child thoughts must be the same or one-to-one should be set to False")
                # connect child thought to all parent thoughts
                for parent in self.parents_thoughts:
                    parent.append_child_operation(self)
                self.children_thoughts[0].parents_operations = [self]
            else:
                if len(self.parents_thoughts)!=len(self.children_thoughts):
                    raise ValueError("The number of parent thoughts and child thoughts must be the same")
                for parent, child in zip(self.parents_thoughts, self.children_thoughts):
                    parent.append_child_operation(self)
                    child.parents_operations = [self]
        else:
            if len(self.children_thoughts)!=1:
                raise ValueError("The number of child thoughts must be one when there is only one parent thought")
            self.parents_thoughts[0].append_child_operation(self)
            self.children_thoughts[0].parents_operations = [self]

    def generate_children(self) -> float:
        self.logger.info(f"Scoring thought: {self.parents_thoughts}")
        cost = 0
        if len(self.children_thoughts)==1:
            if self.scoring_function is not None:
                score = self.scoring_function(self.parents_thoughts[0])
                self.children_thoughts[0].content = score
            else:
                if len(self.parents_thoughts)>1 and not self.is_one_to_one:
                    raise ValueError("Scoring prompt only supports one parent thought when there is more than one parent thought")
                query = self.language_model.query(self.parents_thoughts[0], self.scoring_prompt)
                score = self.language_model.get_response_texts(query)
                self.children_thoughts[0].content = score
                self.children_thoughts[0].is_executable = True
                cost += self.language_model.get_response_cost(query)
        else:
            if self.scoring_function is not None:
                scores = [self.scoring_function(parent) for parent in self.parents_thoughts]
            else:
                for child, parent in zip(self.children_thoughts, self.parents_thoughts):
                    scores = self.language_model.query(parent, self.scoring_prompt)
                    child.content = self.language_model.get_response_texts(scores)
                    child.is_executable = True
                    cost += self.language_model.get_response_cost(scores)
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
            return f"Score(scoring_prompt_name={self.scoring_prompt_name}, children_thoughts={self.children_thoughts})"
    
class Improve(Operation):
    """
    Operation to improve thoughts by querying a language model. (assume the refined thought is better)
    Comparing with ValidateAndImprove, this operation only use the improve_prompt once for each parent thought.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, improve_prompt: Optional['Parser']=None, improve_prompt_name: str="unnamed", children_thoughts: List[Thought]=[]):
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
        if len(children_thoughts)>1 and len(children_thoughts)!=len(parents_thoughts):
            raise ValueError("The number of child thoughts and parent thoughts must be the same")
        self.improve_prompt = improve_prompt
        self.improve_prompt_name = improve_prompt_name
        super().__init__(parents_thoughts, language_model, children_thoughts)

    def connect_thoughts(self) -> None:
        if len(self.children_thoughts)==0:
            self.logger.warning(f"No child thoughts provided, {len(self.parents_thoughts)} new thoughts will be created")
            self.children_thoughts = [Thought(None, [self]) for _ in range(len(self.parents_thoughts))]
        for parent, child in zip(self.parents_thoughts, self.children_thoughts):
            parent.append_child_operation(self)
            child.parents_operations = [self]

    def generate_children(self) -> float:
        self.logger.info(f"Improving thought: {self.parents_thoughts}")
        cost = 0
        for parent, child in zip(self.parents_thoughts, self.children_thoughts):
            query = self.language_model.query(parent, self.improve_prompt)
            child.content = self.language_model.get_response_texts(query)
            cost += self.language_model.get_response_cost(query)
        return cost
    
    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "improve_prompt_name": self.improve_prompt_name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def __str__(self) -> str:
        return f"Improve(improve_prompt_name={self.improve_prompt_name}, children_thoughts={self.children_thoughts})"

class ValidateAndImprove(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(self, parents_thoughts: List[Thought], language_model: AbstractLanguageModel, validate_function: Callable[[Dict], bool]=None, num_tries: int=3, children_thoughts: List[Thought]=[]):
        """
        Initialize a new ValidateAndImprove operation.
        
        :param parents_thoughts: The parent thoughts to validate and improve.
        :type parents_thoughts: List[Thought]
        :param validate_function: A function to validate thoughts (if not using LM).
        :type validate_function: Callable[[Dict], bool]
        :param num_tries: Number of tries to improve the thought before giving up. Defaults to 3.
        :type num_tries: int
        :param children_thoughts: The child thoughts to validate and improve, must have equal length as parents_thoughts.
        :type children_thoughts: List[Thought]
        """
        if validate_function is None:
            raise ValueError("validate_function must be provided")
        self.validate_function = validate_function
        self.num_tries = num_tries
        super().__init__(parents_thoughts, language_model, children_thoughts)

    def generate_children(self) -> None:
        parents=self.parents_thoughts
        for parent in parents:
            current_thought=Thought.from_thought(parent)
            current_try=0
            while True:
                if self.validate_function is not None:
                    valid=self.validate_function(current_thought.state)
                else:
                    prompt=self.prompter.validation_prompt(**current_thought.state)
                    responses=self.language_model.query(prompt, num_responses=self.num_samples)
                    valid=self.parser.parse_validation_answer(current_thought.state, responses)

    def __str__(self) -> str:
        if self.validate_function is not None:
            return f"ValidateAndImprove(validate_function={self.validate_function}, improve={self.improve}, num_tries={self.num_tries})"
        else:
            return f"ValidateAndImprove(validate_prompt={self.validate_prompt}, improve={self.improve}, num_tries={self.num_tries})"
    
    def __json__(self) -> dict:
        if self.validate_function is not None:
            return {
                "operation_type": self.operation_type,
                "validate_function": self.validate_function,
                "improve": self.improve,
                "num_tries": self.num_tries,
            }
        else:
            return {
                "operation_type": self.operation_type,
                "validate_prompt": self.validate_prompt,
                "improve": self.improve,
                "num_tries": self.num_tries,
            }


class Aggregate(Operation):
    """
    Operation to aggregate thoughts.
    """

    operation_type: OperationType = OperationType.aggregate

class KeepBestN(Operation):
    """
    Operation to keep the best N thoughts.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, parents_thoughts: List[Thought], n: int, language_model: AbstractLanguageModel, children_thoughts: List[Thought]=[]):
        super().__init__(parents_thoughts, language_model, children_thoughts)
        self.n = n

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
