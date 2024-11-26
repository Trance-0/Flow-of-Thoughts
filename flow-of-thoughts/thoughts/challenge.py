import logging
from language_models.abstract_language_model import AbstractLanguageModel
from thoughts.thought import Thought

class Challenge():
    def __init__(self, root: Thought, max_budget: float=10, max_thoughts: int=1000):
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
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        self.max_budget = max_budget
        self.max_thoughts = max_thoughts
        self.root = root
        self.logger.debug(f"Challenge initialized with root: {self.root}")

    def run(self) -> None:
        self.logger.debug("Checking that the program is in a valid state")
        assert self.root is not None, "The challenge has no root"

        execution_queue = [
            operation
            for operation in self.root.get_children_operations()
            if operation.is_executable
        ]
        self.logger.debug(f"Execution queue: {[op.operation_type for op in execution_queue]}")
        self.current_budget = 0
        current_thoughts = len(execution_queue)

        while len(execution_queue) > 0:
            if self.current_budget > self.max_budget:
                self.logger.warning("Budget has been depleted, stopping.")
                break
            if current_thoughts > self.max_thoughts:
                self.logger.warning("Number of thoughts has been depleted, stopping.")
                break
            current_operation = execution_queue.pop(0)
            self.logger.info("Executing operation %s", current_operation.operation_type)
            current_operation.generate_children()
            self.logger.info("Operation %s executed", current_operation.operation_type)
            for operation in current_operation.get_children_operations():
                if operation.is_executable:
                    execution_queue.append(operation)
            self.current_budget += current_operation.cost
            current_thoughts += 1
        self.logger.info("All operations executed")

    def __str__(self) -> str:
        return f"Challenge(prompt={self.prompt}, goal={self.goal})"
    
    def __json__(self) -> dict:
        return {
            "max_budget": self.max_budget,
            "max_thoughts": self.max_thoughts,
            "root": self.root.__json__(),
        }

    def get_remaining_budget(self) -> float:
        return self.max_budget - self.current_budget