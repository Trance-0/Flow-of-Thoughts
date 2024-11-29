import logging
from typing import List
from language_models.abstract_language_model import AbstractLanguageModel
from thoughts.thought import Thought
from thoughts.operations import Operation

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

        execution_queue = []
        executed=set()
        for operation in self.root.get_children_operations():
            if isinstance(operation, Operation) and operation.is_executable:
                execution_queue.append(operation)
                executed.add(operation.hash)
        self.logger.debug(f"Execution queue: {[op.operation_type for op in execution_queue]}")
        self.current_budget = 0
        current_thoughts = len(execution_queue)
        current_depth = 0
        while len(execution_queue) > 0:
            layer=len(execution_queue)
            # record if we have executable operation in this layer
            stale=True
            for _ in range(layer):
                if self.current_budget > self.max_budget:
                    self.logger.error(f"Budget has been depleted at depth {current_depth}, stopping.")
                    execution_queue=[]
                    break
                if current_thoughts > self.max_thoughts:
                    self.logger.error("Number of thoughts has been depleted, stopping.")
                    execution_queue=[]
                    break
                current_operation = execution_queue.pop(0)
                if not current_operation.is_executable:
                    execution_queue.append(current_operation)
                    continue
                stale=False
                self.logger.info(f"Executing operation {current_operation.operation_type} at depth {current_depth}, current thought count: {current_thoughts}")
                self.current_budget += current_operation.generate_children()
                self.logger.info(f"Operation {current_operation.operation_type} executed")
                for child in current_operation.get_children_thoughts():
                    for child_op in child.get_children_operations():
                        # only execute operations that all the parents are executed and not executed operations
                        if isinstance(child_op, Operation) and child_op.is_executable and child_op.hash not in executed:
                            execution_queue.append(child_op)
                            executed.add(child_op.hash)
                            current_thoughts += 1
            if stale:
                self.logger.error("No executable operation in layer, stopping.")
                break
            current_depth += 1
        self.logger.info("All operations executed")

    def trace(self,thought_hash: str) -> Thought:
        """
        Trace a thought by its hash. And back to its root.
        """
        return self.root.trace(thought_hash)

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