import logging
from abc import ABC, abstractmethod
from language_models.abstract_language_model import AbstractLanguageModel
from thoughts.thought import Thought

logger = logging.getLogger(__name__)

class Challenge(ABC):
    def __init__(self, root: Thought, prompt: str="", goal: str="", max_budget: float=10, max_thoughts: int=1000):
        """
        Initialize a challenge.

        :param prompt: The prompt for the entire challenge.
        :type prompt: str
        :param language_model: The language model to use for the challenge.
        :type language_model: AbstractLanguageModel
        :param goal: The goal of the challenge.
        :type goal: str
        :param max_budget: The maximum budget for the challenge.
        :type max_budget: float
        :param max_thoughts: The maximum number of thoughts for the challenge.
        :type max_thoughts: int
        """
        self.prompt = prompt
        self.goal = goal
        self.max_budget = max_budget
        self.max_thoughts = max_thoughts
        self.root = root

    def run(self) -> None:
        logger.debug("Checking that the program is in a valid state")
        assert self.root is not None, "The challenge has no root"
        logger.debug("The program is in a valid state")

        execution_queue = [
            operation
            for operation in self.root.children
            if operation.is_executable
        ]
        current_budget = 0
        current_thoughts = len(execution_queue)

        while len(execution_queue) > 0:
            if current_budget > self.max_budget:
                logger.warning("Budget has been depleted, stopping.")
                break
            if current_thoughts > self.max_thoughts:
                logger.warning("Number of thoughts has been depleted, stopping.")
                break
            current_operation = execution_queue.pop(0)
            logger.info("Executing operation %s", current_operation.operation_type)
            current_operation.generate_child()
            logger.info("Operation %s executed", current_operation.operation_type)
            for operation in current_operation.child_thought:
                if operation.is_executable:
                    execution_queue.append(operation)
            current_budget += current_operation.cost
            current_thoughts += 1
        logger.info("All operations executed")
        self.run_executed = True

    def __str__(self) -> str:
        return f"Challenge(prompt={self.prompt}, goal={self.goal})"
    
    def __json__(self) -> dict:
        return {
            "prompt": self.prompt,
            "goal": self.goal,
            "root": self.root.__json__(),
        }
