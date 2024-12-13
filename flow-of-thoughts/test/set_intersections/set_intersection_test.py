from collections import defaultdict
import csv
import datetime
import json
import logging
import os
from typing import List
from language_models.chatgpt import ChatGPT
from language_models.claude import Claude
from thoughts.thought import Thought
from thoughts.operations import (
    Aggregate,
    Conditional,
    Generate,
    Evaluate,
    Improve,
    KeepBestN,
    Link,
    Parser,
    Score,
    Split,
)
from thoughts.challenge import Challenge

budget = 30

# use universal language model
lm_name = "chatgpt4o-mini"
lm = ChatGPT(
    model_name=lm_name,
    cache=True,
)

# lm_name = "claude-3.5-sonnet"
# lm = Claude(
#     model_name=lm_name,
#     cache=True,
# )

prompts = None

# load prompts
with open(os.path.join(os.path.dirname(__file__), "set_intersection_prompt.json"), "r") as f:
    prompts = json.load(f)

def str_to_list(s: str) -> list:
    return [int(x) for x in s[1:-1].split(",")]

def num_errors(thought: Thought, original_thought: Thought) -> int:
    """
    Calculate the number of errors in the intersection.
    """
    y = set(str_to_list(thought.content))
    y_truth = set(str_to_list(original_thought.content))
    return len(y.symmetric_difference(y_truth))

def compare_intersection(thought: Thought, truth: str) -> float:
    """
    Calculate the percentage of errors in the intersection.
    """
    y = set(str_to_list(thought.content))
    y_truth = set(str_to_list(truth))
    if len(y_truth) == 0:
        return 1.0 if len(y) > 0 else 0.0
    return len(y.symmetric_difference(y_truth)) / (len(y.union(y_truth)))

def io(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the IO method.
    """
    root = Thought(task_input, [], [], is_executable=True)
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "set_intersection_prompt")
    )
    res = Evaluate(
        generate.get_children_thoughts(),
        lm,
        evaluate_function=compare_intersection,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def tot(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the TOT method.
    """
    root = Thought(task_input, [], [], is_executable=True)
    generate = Generate(
        [root], lm, 5, generate_prompt=Parser.from_json(prompts, "set_intersection_prompt")
    )
    scoring_layer = [
        Score([t], lm, scoring_function=num_errors, original_thought=root)
        for t in generate.get_children_thoughts()
    ]
    keep_best = KeepBestN([s.get_children_thoughts()[0] for s in scoring_layer], lm, 1, False)
    improve = Improve(
        keep_best.get_children_thoughts(),
        lm,
        5,
        original_thought=root,
        improve_prompt=Parser.from_json(prompts, "tot_improve_prompt"),
    )
    res = Evaluate(
        improve.get_children_thoughts(),
        lm,
        evaluate_function=compare_intersection,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def fot(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the FOT method.
    """
    root = Thought("Generate methods to find set intersection", [], [], is_executable=True)
    task_thought = Thought(task_input, [], [], is_executable=False)
    link_operation = Link([root], lm, [task_thought])
    generate_method = Generate(
        [root],
        lm,
        1,
        generate_prompt=Parser.from_json(prompts, "fot_generate_prompt"),
    )
    parse_methods = Split([generate_method.get_children_thoughts()[0]], lm, ["Method 1", "Method 2"])
    methods_thoughts = []
    
    for method_thought in parse_methods.get_children_thoughts():
        pits_gen_thought = Thought(f"Generate pitfalls for: {method_thought.content}", [], [], is_executable=True)
        link_operation = Link([method_thought], lm, [pits_gen_thought])
        pits = Generate(
            [pits_gen_thought],
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "fot_check_prompt"),
        )
        parse_pits = Split([pits.get_children_thoughts()[0]], lm, ["Pitfall 1", "Pitfall 2", "Pitfall 3"])
        
        ego = Generate(
            [method_thought],
            lm,
            1,
            generate_prompt=Parser(prompt=f"Use this method to find intersection: {method_thought.content}\nInput: {task_input}", name="use_method_prompt", plain=True),
        )
        
        for pitfall in parse_pits.get_children_thoughts():
            improve_parser = Parser.from_json(prompts, "fot_improve_prompt")
            improve_parser.cot = pitfall.content
            link_operation = Link([pitfall], lm, [ego.get_children_thoughts()[0]])
            ego = Improve(
                link_operation.get_children_thoughts(),
                lm,
                1,
                improve_prompt=improve_parser,
                original_thought=task_thought,
            )
        methods_thoughts.append(ego.get_children_thoughts()[0])
    
    final_scores = [
        Score(
            [method],
            lm,
            scoring_function=num_errors,
            original_thought=task_thought
        ).get_children_thoughts()[0]
        for method in methods_thoughts
    ]
    keep_best = KeepBestN(final_scores, lm, 1, False)
    res = Evaluate(
        keep_best.get_children_thoughts(),
        lm,
        evaluate_function=compare_intersection,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def set_intersection_064():
    global budget
    data_ids = [0]
    methods = [io, tot, fot]
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "set_intersection_064.csv")

    data = []
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([int(row[0]), row[1], row[2]])

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    final_results = defaultdict(list)
    for data in selected_data:
        logging.info(f"Running data {data[0]}: {data[1]}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
            logging.info(f"Running method {method.__name__}")
            logging.info(f"Budget left: {budget}")
            if budget <= 0.0:
                logging.error(
                    f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                )
                break
            cur_challenge = method(data[1], data[2])
            try:
                cur_challenge.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}_{method.__name__}_{timestamp}.json",
            )
            res = cur_challenge.as_G6_graph()
            with open(path, "w") as f:
                json.dump(res, f)
            budget = cur_challenge.get_remaining_budget()
            final_results[method.__name__].append(cur_challenge.get_final_accuracy())
    logging.info(f"budget remains {budget}")
    for method in methods:
        logging.info(f"{method.__name__}: {final_results[method.__name__]}")
