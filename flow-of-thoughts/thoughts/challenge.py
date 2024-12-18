import logging
from typing import List, Tuple
from language_models.abstract_language_model import AbstractLanguageModel
from thoughts.thought import Thought
from thoughts.operations import Operation, OperationType

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

    def get_graph(self) -> Tuple[List[Thought],List[Operation]]:
        # get the graph of the challenge with edge and vertex
        thoughts = set([self.root])
        operations = set()
        search_queue = [self.root]
        while len(search_queue) > 0:
            current = search_queue.pop(0)
            for operation in current.get_children_operations():
                if operation not in operations:
                    operations.add(operation)
                for child in operation.get_children_thoughts():
                    if child not in thoughts:   
                        thoughts.add(child)
                        search_queue.append(child)
        return [list(thoughts), list(operations)]
    
    def get_final_accuracy(self,numerical_accuracy: bool=False) -> float|tuple[str,str,float]:
        """
        Get the final accuracy of the challenge.
        
        :param numerical_accuracy: Whether to return the numerical accuracy or the accuracy in the form of a string.
        :type numerical_accuracy: bool
        :return: The final accuracy of the challenge. (numerical_accuracy)|(result,expected,accuracy)
        :rtype: float|tuple[str,str,float]
        """
        thoughts, operations = self.get_graph()
        # we don't care about the duplicates
        operation_dict={op.operation_type: op for op in operations}
        if numerical_accuracy:
            return float(operation_dict[OperationType.evaluate].get_children_thoughts()[0].content) if operation_dict[OperationType.evaluate].get_children_thoughts()[0].content!='' else -1
        else:
            return operation_dict[OperationType.evaluate].get_parents_thoughts()[0].content, operation_dict[OperationType.evaluate].ground_truth, float(operation_dict[OperationType.evaluate].get_children_thoughts()[0].content) if operation_dict[OperationType.evaluate].get_children_thoughts()[0].content!='' else -1
    
    def __str__(self) -> str:
        return f"Challenge(max_budget={self.max_budget}, max_thoughts={self.max_thoughts})"
    
    def as_G6_graph(self) -> dict:
        thoughts, operations = self.get_graph()
        nodes=[thought.as_G6_node() for thought in thoughts]
        edges=[]
        for operation in operations:
            edges.extend(operation.as_G6_edges())
        return {
            "nodes": nodes,
            "edges": edges,
        }
    
    def __json__(self) -> dict:
        graph = self.get_graph()
        return {
            "max_budget": self.max_budget,
            "max_thoughts": self.max_thoughts,
            "total_thoughts_count": len(graph[0]),
            "total_operations_count": len(graph[1]),
            "graph": {"nodes": [thought.__json__() for thought in graph[0]], "edges": [operation.__json__() for operation in graph[1]]},
        }

    def get_remaining_budget(self) -> float:
        return self.max_budget - self.current_budget