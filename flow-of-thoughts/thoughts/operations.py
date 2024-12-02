from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum
import itertools
import json
import random
from typing import Callable, Dict, Iterator, List, Optional, Any, Tuple
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
    split: int = 6
    keep_best_n: int = 7
    conditional: int = 8
    retrieve: int = 9

    def __str__(self) -> str:
        return self.name


class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        children_thoughts: Optional[List[Thought]] = None,
        input_size: int = -1,
        output_size: int = -1,
    ):
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
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        # validate input size and output size
        if input_size < 1:
            raise ValueError("The input size must be greater than 0")
        if output_size < 1:
            output_size = input_size
        self.hash = random.randint(0, 2**64 - 1)
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
        if self.get_children_thoughts is None or len(self.get_children_thoughts()) == 0:
            if self.output_size == 1:
                self.logger.warning(
                    "No child thoughts provided,a new thought will be created"
                )
                self.children_thoughts = [Thought("", [self], [])]
            else:
                self.logger.warning(
                    f"No child thoughts provided, {self.output_size} new thoughts will be created"
                )
                self.children_thoughts = [
                    Thought("", [self], []) for _ in range(self.output_size)
                ]
        # connect thoughts with operation
        self.logger.info(f"Connecting {self.input_size} parent thoughts to {self}")
        for parent in self.get_parents_thoughts():
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
        return self.children_thoughts if self.children_thoughts is not None else []

    def get_parents_thoughts(self) -> List[Thought]:
        return self.parents_thoughts if self.parents_thoughts is not None else []

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __full_json__(self) -> dict:
        pass

    @abstractmethod
    def as_G6_edges(self) -> List[dict]:
        pass

    @abstractmethod
    def __json__(self) -> dict:
        pass


class Parser:
    """
    Class for parsing the input and output of a language model.
    """

    def __init__(
        self,
        prompt: str,
        cot: str = None,
        shots: List[dict] = None,
        name: str = "unnamed",
        plain=False,
    ):
        """
        Initialize a parser.

        :param prompt: The prompt for the parser.
        :type prompt: str
        :param cot: The cot for the parser.
        :type cot: str
        :param shots: The shots for the parser.
        :type shots: List[dict]
        :param plain: Whether the parser is plain, if true, the prompt will be returned as is without any additional formatting.
        :type plain: bool
        """
        self.prompt = prompt
        self.cot = cot
        self.shots = shots
        self.name = name
        self.plain = plain
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def parse(self, input: str) -> str:
        """
        Parse the input to a string that can be used as a query to a language model.
        """
        if self.plain:
            return self.prompt
        res = f"<Instruction> {self.prompt} </Instruction>"
        if self.cot is not None and len(self.cot) > 0:
            res += f"\n<Approach>\n{self.cot}\n</Approach>"
        if self.shots is not None and len(self.shots) > 0:
            partial_examples = []
            for shot in self.shots:
                current_example = ""
                current_example += f"Input: {shot['input']}\n"
                for key, value in shot.items():
                    if key != "input" and key != "output":
                        current_example += f"{key}: {value}\n"
                current_example += f"Output: {shot['output']}"
                partial_examples.append(current_example)
            res += "\n<Examples>\n" + "\n".join(partial_examples) + "\n</Examples>"
        res = res + f"\nInput: {input} \nOutput:"
        self.logger.info(f"Parser input: {input}, final prompt: {res}")
        return res

    @staticmethod
    def from_dict(prompt: dict) -> "Parser":
        """
        Initialize a parser from a dictionary.
        """
        return Parser(
            prompt["instruction"],
            prompt.get("approach", ""),
            prompt.get("examples", []),
            name=prompt.get("name", "unnamed"),
        )

    @staticmethod
    def from_json(json_data: dict, name: str = "unnamed") -> "Parser":
        """
        Initialize a parser from a json dictionary.
        """
        content = json_data[name]
        parser = Parser.from_dict(content)
        parser.name = name
        return parser

    def __str__(self) -> str:
        return f"Parser(prompt={self.prompt}, cot={self.cot}, shots={self.shots}, name={self.name})"

    def __json__(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "cot": self.cot,
            "shots": self.shots,
        }


class Generate(Operation):
    """
    Operation to generate thoughts.

    Input: one thought to generate children from
    Output: children thoughts
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        branching_factor: int,
        generate_prompt: Parser = None,
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Generate operation.

        :param parents_thoughts: The parent thought to generate children from, in our implementation, the length of the parent thoughts must be one.
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for generation.
        :type language_model: AbstractLanguageModel
        :param branching_factor: The branching factor for generation, the number of children thoughts to generate for each parent thought, equiv to num_responses in language model query.
        :type branching_factor: int
        :param generate_prompt_name: The name of the generate prompt when generating json.
        :type generate_prompt_name: str
        :param children_thoughts: The child thoughts to generate.
        :type children_thoughts: List[Thought]
        """
        self.branching_factor = branching_factor
        if generate_prompt is None:
            raise ValueError("generate_prompt must be provided")
        self.generate_prompt = generate_prompt
        # default name for unnamed generate prompt
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=branching_factor,
        )

    def generate_children(self) -> float:
        self.logger.info(
            f"Generating {self.branching_factor} children for {self.parents_thoughts}"
        )
        query = None
        # if generate multiple children thoughts
        if self.branching_factor > 1:
            self.logger.info(
                f"Generating {self.branching_factor} children for {self.parents_thoughts} with different tries"
            )
            query = self.language_model.query(
                self.generate_prompt.parse(self.parents_thoughts[0].content),
                num_responses=self.branching_factor,
            )
            response_texts = self.language_model.get_response_texts(query)
            for i in range(self.branching_factor):
                child = self.children_thoughts[i]
                child.content = response_texts[i]
                child.is_executable = True
        else:
            query = self.language_model.query(
                self.generate_prompt.parse(self.parents_thoughts[0].content)
            )
            child = self.children_thoughts[0]
            child.content = self.language_model.get_response_texts(query)[0]
            child.is_executable = True
        return self.language_model.get_response_cost(query)

    def __str__(self) -> str:
        return f"Generate(generate_prompt_name={self.generate_prompt.name}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "generate_prompt_name": self.generate_prompt.name,
            "children_thoughts": [
                child.__json__() for child in self.get_children_thoughts()
            ],
        }

    def as_G6_edges(self) -> List[dict]:
        res = []
        for child in self.get_children_thoughts():
            res.append(
                {
                    "source": hex(self.get_parents_thoughts()[0].hash),
                    "target": hex(child.hash),
                    "label": self.generate_prompt.name,
                }
            )
        return res

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "id": hex(self.hash),
            "generate_prompt_name": self.generate_prompt.name,
            "children_thoughts": [
                hex(child.hash) for child in self.get_children_thoughts()
            ],
        }


class Evaluate(Operation):
    """
    Operation to evaluate thoughts by comparing the output of the thought with the truth.
    """

    operation_type: OperationType = OperationType.evaluate

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        evaluate_prompt: Optional["Parser"] = None,
        evaluate_function: Callable[[Thought, Any], float] = None,
        ground_truth: Any = None,
        children_thoughts: List[Thought] = None,
    ):
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
            raise ValueError(
                "Either evaluate_function or evaluate_prompt must be provided"
            )
        if (evaluate_function is not None) and (evaluate_prompt is not None):
            raise ValueError(
                "Only one of evaluate_function or evaluate_prompt can be provided"
            )
        self.evaluate_function = evaluate_function
        if evaluate_function is not None and ground_truth is None:
            raise ValueError(
                "ground_truth must be provided when evaluate_function is used"
            )
        self.ground_truth = ground_truth
        self.evaluate_prompt = evaluate_prompt
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=1,
        )

    def generate_children(self) -> float:
        self.logger.info(f"Evaluating thought: {self.parents_thoughts}")
        cost = 0
        score = None
        if self.evaluate_function is not None:
            score = self.evaluate_function(self.parents_thoughts[0], self.ground_truth)
        else:
            query = self.language_model.query(
                self.evaluate_prompt.parse(self.parents_thoughts[0].content)
            )
            score = float(self.language_model.get_response_texts(query)[0])
            cost += self.language_model.get_response_cost(query)
        self.children_thoughts[0].content = score
        self.children_thoughts[0].is_executable = True
        return cost

    def __str__(self) -> str:
        if self.evaluate_function is not None:
            return f"Evaluate(evaluate_function={self.evaluate_function}, ground_truth={self.ground_truth}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"
        else:
            return f"Evaluate(evaluate_prompt_name={self.evaluate_prompt.name}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        if self.evaluate_function is not None:
            return {
                "operation_type": self.operation_type,
                "evaluate_function": self.evaluate_function.__name__,
                "ground_truth": self.ground_truth,
                "children_thoughts": [
                    child.__json__() for child in self.children_thoughts
                ],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "evaluate_prompt_name": self.evaluate_prompt.name,
                "ground_truth": self.ground_truth,
                "children_thoughts": [
                    child.__json__() for child in self.children_thoughts
                ],
            }

    def as_G6_edges(self) -> List[dict]:
        return [
            {
                "source": hex(self.get_parents_thoughts()[0].hash),
                "target": hex(self.get_children_thoughts()[0].hash),
                "label": (
                    self.evaluate_prompt.name
                    if self.evaluate_prompt is not None
                    else self.evaluate_function.__name__
                ),
            }
        ]

    def __json__(self) -> dict:
        if self.evaluate_function is not None:
            return {
                "operation_type": self.operation_type,
                "hash": self.hash,
                "evaluate_function": self.evaluate_function.__name__,
                "ground_truth": self.ground_truth,
                "children_thoughts": [
                    child.hash for child in self.get_children_thoughts()
                ],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "hash": self.hash,
                "evaluate_prompt_name": self.evaluate_prompt.name,
                "ground_truth": self.ground_truth,
                "children_thoughts": [
                    child.hash for child in self.get_children_thoughts()
                ],
            }


class Score(Operation):
    """
    Operation to score thoughts.

    Input: thoughts
    Output: thoughts with scores
    """

    operation_type: OperationType = OperationType.score

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        original_thought: Thought,
        scoring_function: Callable[[Thought, Thought], float] = None,
        scoring_prompt: Optional["Parser"] = None,
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Score operation.

        :param parents_thoughts: The parent thoughts to score length must be 1, will only generate one child thought.
        :type parents_thoughts: List[Thought]
        :param language_model: The language model to use for scoring.
        :type language_model: AbstractLanguageModel
        :param original_thought: The original thought to score.
        :type original_thought: Thought
        :param scoring_function: A function to score thoughts.
        :type scoring_function: Callable[[Thought, Thought], float]
        :param scoring_prompt: A parser to score thoughts.
        :type scoring_prompt: Parser
        :param children_thoughts: The child thoughts to score length must be 1, will only generate one child thought.
        :type children_thoughts: List[Thought]
        """
        self.original_thought = original_thought
        self.scoring_function = scoring_function
        self.scoring_prompt = scoring_prompt
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=1,
        )

    def generate_children(self) -> float:
        self.logger.info(f"Scoring thought: {self.parents_thoughts}")
        cost = 0
        if self.scoring_function is not None:
            score = self.scoring_function(
                self.parents_thoughts[0], self.original_thought
            )
        else:
            query = self.language_model.query(
                f"Input:{self.scoring_prompt.parse(self.parents_thoughts[0].content)}\n original_thought:{self.original_thought.content}"
            )
            score = float(self.language_model.get_response_texts(query)[0])
            cost += self.language_model.get_response_cost(query)
        self.children_thoughts[0].content = score
        self.children_thoughts[0].is_executable = True
        return cost

    def __str__(self) -> str:
        if self.scoring_function is not None:
            return f"Score(scoring_function={self.scoring_function}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"
        else:
            return f"Score(scoring_prompt_name={self.scoring_prompt.name}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        if self.scoring_function is not None:
            return {
                "operation_type": self.operation_type,
                "scoring_function": self.scoring_function.__name__,
                "children_thoughts": [
                    child.__json__() for child in self.children_thoughts
                ],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "scoring_prompt_name": self.scoring_prompt.name,
                "children_thoughts": [
                    child.__json__() for child in self.children_thoughts
                ],
            }

    def as_G6_edges(self) -> List[dict]:
        return [
            {
                "source":hex(self.original_thought.hash),
                "target":hex(self.get_children_thoughts()[0].hash),
                "label": "original_thought",
            },
            {
                "source": hex(self.get_parents_thoughts()[0].hash),
                "target": hex(self.get_children_thoughts()[0].hash),
                "label": (
                    self.scoring_prompt.name
                    if self.scoring_prompt is not None
                    else self.scoring_function.__name__
                ),
            }
        ]

    def __json__(self) -> dict:
        if self.scoring_function is not None:
            return {
                "operation_type": self.operation_type,
                "hash": self.hash,
                "original_thought_content": self.original_thought.content,
                "scoring_function": self.scoring_function.__name__,
                "children_thoughts": [
                    child.hash for child in self.get_children_thoughts()
                ],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "hash": self.hash,
                "original_thought_content": self.original_thought.content,
                "scoring_prompt_name": self.scoring_prompt.name,
                "children_thoughts": [
                    child.hash for child in self.get_children_thoughts()
                ],
            }


class Improve(Operation):
    """
    Operation to improve thoughts by querying a language model. (assume the refined thought is better)
    Comparing with ValidateAndImprove, this operation only use the improve_prompt once for each parent thought.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        branching_factor: int,
        original_thought: Thought,
        improve_prompt: Parser,
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Improve operation.

        :param parents_thoughts: The parent thoughts to improve.
        :type parents_thoughts: List[Thought]
        :param branching_factor: The number of children thoughts to generate same as the branching factor of the Generate operation.
        :type branching_factor: int
        :param improve_prompt: A parser to improve thoughts.
        :type improve_prompt: Parser
        :param improve_prompt_name: The name of the improve prompt when generating json.
        :type improve_prompt_name: str
        :param children_thoughts: The child thoughts to improve.
        :type children_thoughts: List[Thought]
        """
        self.branching_factor = branching_factor
        self.improve_prompt = improve_prompt
        self.original_thought = original_thought
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=branching_factor,
        )

    def generate_children(self) -> float:
        self.logger.info(f"Improving thought: {self.get_parents_thoughts()}")
        parent_thought = self.get_parents_thoughts()[0]
        original_thought = self.original_thought
        query = self.language_model.query(
            self.improve_prompt.parse(
                f"Input: {original_thought.content}, \n Incorrect output: {parent_thought.content}"
            ),
            num_responses=self.branching_factor,
        )
        for i in range(self.branching_factor):
            self.children_thoughts[i].content = self.language_model.get_response_texts(
                query
            )[i]
            self.children_thoughts[i].is_executable = True
        return self.language_model.get_response_cost(query)

    def __str__(self) -> str:
        return f"Improve(improve_prompt_name={self.improve_prompt.name}, original_thought_content={self.original_thought.content}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "improve_prompt_name": self.improve_prompt.name,
            "original_thoughts_content": self.original_thought.content,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def as_G6_edges(self) -> List[dict]:
        res = []
        for child in self.get_children_thoughts():
            res.append(
                {
                    "source": hex(self.get_parents_thoughts()[0].hash),
                    "target": hex(child.hash),
                    "label": self.improve_prompt.name,
                }
            )
            res.append(
                {
                    "source": hex(self.original_thought.hash),
                    "target": hex(child.hash),
                    "label": "original_thought",
                }
            )
        return res

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "hash": self.hash,
            "improve_prompt_name": self.improve_prompt.name,
            "original_thought_content": self.original_thought.content,
            "children_thoughts": [child.hash for child in self.get_children_thoughts()],
        }


class Validate(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        validate_function: Callable[[str], bool] = None,
        validate_prompt: Parser = None,
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Validate operation.

        :param parents_thoughts: The parent thoughts to validate and improve.
        :type parents_thoughts: List[Thought]
        :param validate_function: A function to validate thoughts (if not using LM).
        :type validate_function: Callable[[str], bool]
        :param validate_prompt: A parser to validate thoughts.
        :type validate_prompt: Parser
        :param children_thoughts: The child thoughts to validate and improve, must have equal length as parents_thoughts.
        :type children_thoughts: List[Thought]
        """
        if validate_function is None and validate_prompt is None:
            raise ValueError("validate_function or validate_prompt must be provided")
        if validate_function is not None and validate_prompt is not None:
            raise ValueError(
                "validate_function and validate_prompt cannot be provided at the same time"
            )
        self.validate_function = validate_function
        self.validate_prompt = validate_prompt
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=1,
        )

    def generate_children(self) -> float:
        parent = self.parents_thoughts[0]
        cost = 0
        if self.validate_function is not None:
            self.children_thoughts[0].content = self.validate_function(parent.content)
        else:
            query = self.language_model.query(
                self.validate_prompt.parse(parent.content)
            )
            cost += self.language_model.get_response_cost(query)
            self.children_thoughts[0].content = self.language_model.get_response_texts(
                query
            )[0]
        self.children_thoughts[0].is_executable = True
        self.logger.info(f"Validated thought: {self.children_thoughts}")
        return cost

    def __str__(self) -> str:
        if self.validate_function is not None:
            return f"Validate(validate_function={self.validate_function.__name__})"
        else:
            return f"Validate(validate_prompt_name={self.validate_prompt.name})"

    def __full_json__(self) -> dict:
        if self.validate_function is not None:
            return {
                "operation_type": self.operation_type,
                "validate_function": self.validate_function.__name__,
                "children_thoughts": [
                    child.__json__() for child in self.children_thoughts
                ],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "validate_prompt_name": self.validate_prompt.name,
                "children_thoughts": [
                    child.__json__() for child in self.children_thoughts
                ],
            }

    def as_G6_edges(self) -> List[dict]:
        return [
            {
                "source": hex(self.get_parents_thoughts()[0].hash),
                "target": hex(self.get_children_thoughts()[0].hash),
                "label": (
                    self.validate_prompt.name
                    if self.validate_prompt is not None
                    else self.validate_function.__name__
                ),
            }
        ]

    def __json__(self) -> dict:
        if self.validate_function is not None:
            return {
                "operation_type": self.operation_type,
                "hash": self.hash,
                "validate_function": self.validate_function.__name__,
                "children_thoughts": [
                    child.hash for child in self.get_children_thoughts()
                ],
            }
        else:
            return {
                "operation_type": self.operation_type,
                "hash": self.hash,
                "validate_prompt_name": self.validate_prompt.name,
                "children_thoughts": [
                    child.hash for child in self.get_children_thoughts()
                ],
            }


class Aggregate(Operation):
    """
    Operation to aggregate thoughts.
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        branching_factor: int,
        aggregate_prompt: Optional["Parser"],
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Aggregate operation.

        :param parents_thoughts: The parent thoughts to aggregate.
        :type parents_thoughts: List[Thought]
        :param branching_factor: The number of children thoughts to generate.
        :type branching_factor: int
        :param aggregate_prompt: A parser to aggregate thoughts.
        :type aggregate_prompt: Parser
        :param children_thoughts: The child thoughts to aggregate, must have equal length as parents_thoughts.
        :type children_thoughts: List[Thought]
        """
        if aggregate_prompt is None:
            raise ValueError("aggregate_prompt must be provided")
        self.aggregate_prompt = aggregate_prompt
        self.branching_factor = branching_factor
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=2,
            output_size=branching_factor,
        )

    def generate_children(self) -> float:
        self.logger.info(f"Aggregating thoughts: {self.parents_thoughts}")
        query = self.language_model.query(
            self.aggregate_prompt.parse(
                "\n".join(
                    [
                        f"Input {i+1}: {parent.content}"
                        for i, parent in enumerate(self.parents_thoughts)
                    ]
                )
            ),
            num_responses=self.branching_factor,
        )
        for i in range(self.branching_factor):
            self.children_thoughts[i].content = self.language_model.get_response_texts(
                query
            )[i]
            self.children_thoughts[i].is_executable = True
        return self.language_model.get_response_cost(query)

    def __str__(self) -> str:
        return f"Aggregate(aggregate_prompt_name={self.aggregate_prompt.name}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "aggregate_prompt_name": self.aggregate_prompt.name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def as_G6_edges(self) -> List[Dict]:
        res = []
        for parent in self.get_parents_thoughts():
            for child in self.get_children_thoughts():
                res.append(
                    {
                        "source": hex(parent.hash),
                        "target": hex(child.hash),
                        "label": self.aggregate_prompt.name,
                    }
                )
        return res

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "hash": self.hash,
            "aggregate_prompt_name": self.aggregate_prompt.name,
            "children_thoughts": [child.hash for child in self.get_children_thoughts()],
        }


class Split(Operation):
    """
    Operation to split thoughts.
    """

    operation_type: OperationType = OperationType.split

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        split_key: List[str],
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Split operation.
        """
        self.split_key = split_key
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=len(split_key),
        )

    def generate_children(self) -> float:
        self.logger.info(f"Splitting thought: {self.parents_thoughts}")
        parent_content = defaultdict(str)
        try:
            parent_content = json.loads(self.parents_thoughts[0].content)
        except:
            self.logger.error(
                "Parent thought content is not a valid JSON format, please check the prompt or the thought content"
            )
        for i in range(len(self.split_key)):
            self.children_thoughts[i].content = parent_content[self.split_key[i]]
            self.children_thoughts[i].is_executable = True
            self.children_thoughts[i].append_parent_operation(self.parents_thoughts[0])
        return 0

    def __str__(self) -> str:
        return f"Split(split_key={self.split_key}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "split_key": self.split_key,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def as_G6_edges(self) -> List[Dict]:
        res = []
        for index, child in enumerate(self.get_children_thoughts()):
            res.append(
                {
                    "source": hex(self.get_parents_thoughts()[0].hash),
                    "target": hex(child.hash),
                    "label": self.split_key[index],
                }
            )
        return res

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "hash": self.hash,
            "split_key": self.split_key,
            "children_thoughts": [child.hash for child in self.get_children_thoughts()],
        }


class KeepBestN(Operation):
    """
    Operation to keep the best N thoughts. Must score the thoughts before proceeding this operation.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        n: int,
        highest_score_first: bool = True,
        children_thoughts: List[Thought] = None,
    ):
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
            if (
                len(parent.get_parents_operations()) != 1
                or parent.get_parents_operations()[0].operation_type
                != OperationType.score
            ):
                raise ValueError("All parent thoughts must be scored")
        self.n = n
        self.highest_score_first = highest_score_first
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=len(parents_thoughts),
            output_size=n,
        )

    def generate_children(self) -> float:
        self.logger.info(f"Keeping best {self.n} thoughts: {self.parents_thoughts}")
        q = []
        for parent in self.get_parents_thoughts():
            q.append((float(parent.content), parent))
        q.sort(key=lambda x: x[0], reverse=self.highest_score_first)
        self.logger.info(f"First 10 sorted thoughts: {q[:10]}")
        for i in range(self.n):
            # going up two levels, ignoring the parent scoring layer
            self.children_thoughts[i].content = (
                q[i][1].get_parents_operations()[0].get_parents_thoughts()[0].content
            )
            self.children_thoughts[i].is_executable = True
        return 0

    def __str__(self) -> str:
        return f"KeepBestN(n={self.n}, highest_score_first={self.highest_score_first}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "n": self.n,
            "highest_score_first": self.highest_score_first,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def as_G6_edges(self) -> List[Dict]:
        res = []
        for child in self.get_children_thoughts():
            for parent in self.get_parents_thoughts():
                res.append(
                    {
                        "source": hex(parent.hash),
                        "target": hex(child.hash),
                        "label": f"KeepBestN({self.n})",
                    }
                )
        return res

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "hash": self.hash,
            "n": self.n,
            "highest_score_first": self.highest_score_first,
            "children_thoughts": [child.hash for child in self.get_children_thoughts()],
        }


class Conditional(Operation):
    """
    Operation to condition thoughts.
    """

    operation_type: OperationType = OperationType.conditional

    def __init__(
        self,
        parents_thoughts: List[Thought],
        language_model: AbstractLanguageModel,
        conditional_thoughts: List[Thought],
        condition_function: Callable[[str], bool] = None,
        conditional_prompt: Parser = None,
        children_thoughts: List[Thought] = None,
    ):
        """
        Initialize a new Conditional operation.
        """
        if len(conditional_thoughts) < 1:
            raise ValueError("conditional_thoughts must have at least one thought")
        self.conditional_thoughts = conditional_thoughts
        if condition_function is None and conditional_prompt is None:
            raise ValueError(
                "condition_function or conditional_prompt must be provided"
            )
        if condition_function is not None and conditional_prompt is not None:
            raise ValueError(
                "condition_function and conditional_prompt cannot be provided at the same time"
            )
        self.condition_function = condition_function
        self.conditional_prompt = conditional_prompt
        super().__init__(
            parents_thoughts,
            language_model,
            children_thoughts,
            input_size=1,
            output_size=1,
        )

    def __check_condition(self) -> Tuple[bool, float]:
        """
        Check if the condition is met.

        :return: A tuple of (condition_met, cost)
        :rtype: Tuple[bool, float]
        """
        if self.condition_function is not None:
            return self.condition_function(self.conditional_thoughts), 0
        else:
            query = self.language_model.query(
                self.conditional_prompt.parse(self.parents_thoughts[0].content)
            )
            return float(
                self.language_model.get_response_texts(query)[0]
            ), self.language_model.get_response_cost(query)

    def generate_children(self) -> float:
        self.logger.info(f"Conditioning thought: {self.parents_thoughts}")
        condition_met, cost = self.__check_condition()
        if condition_met:
            self.logger.info("Condition met, executing thought")
            query = self.language_model.query(
                self.conditional_prompt.parse(self.parents_thoughts[0].content)
            )
            self.children_thoughts[0].content = self.language_model.get_response_texts(
                query
            )[0]
            self.children_thoughts[0].is_executable = True
            return cost + self.language_model.get_response_cost(query)
        else:
            self.logger.info("Condition not met, skipping execution")
            self.children_thoughts[0].content = self.parents_thoughts[0].content
            return cost

    def __str__(self) -> str:
        return f"Conditional(conditional_prompt_name={self.conditional_prompt.name}, children_thoughts=[{len(self.get_children_thoughts())}]={','.join([str(child) for child in self.get_children_thoughts()])})"

    def __full_json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "conditional_prompt_name": self.conditional_prompt.name,
            "children_thoughts": [child.__json__() for child in self.children_thoughts],
        }

    def as_G6_edges(self) -> List[Dict]:
        res = [
            {
                "source": hex(self.get_children_thoughts()[0].hash),
                "target": hex(self.get_parents_thoughts()[0].hash),
                "label": (
                    self.conditional_prompt.name
                    if self.condition_function is None
                    else self.condition_function.__name__
                ),
            }
        ]
        for conditional_thought in self.conditional_thoughts:
            res.append(
                {
                    "source": hex(conditional_thought.hash),
                    "target": hex(self.get_children_thoughts()[0].hash),
                    "label": "Condition",
                }
            )
        return res

    def __json__(self) -> dict:
        return {
            "operation_type": self.operation_type,
            "hash": self.hash,
            "conditional_prompt_name": self.conditional_prompt.name,
            "children_thoughts": [child.hash for child in self.get_children_thoughts()],
        }
