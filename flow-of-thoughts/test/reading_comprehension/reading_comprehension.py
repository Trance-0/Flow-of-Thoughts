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

prompts = None

# load prompts
with open(os.path.join(os.path.dirname(__file__), "reading_comprehension_prompt.json"), "r") as f:
    prompts = json.load(f)

def num_errors(thought: Thought, original_thought: Thought) -> int:
    """
    Calculate the number of errors in the answers.

    :param thought: The thought to evaluate
    :type thought: Thought
    :param original_thought: The original thought containing correct answers
    :type original_thought: Thought
    :return: The number of errors in the answers
    :rtype: int
    """
    y = thought.content.split(",")
    y_truth = original_thought.content.split(",")
    if len(y) != len(y_truth):
        if len(y) < len(y_truth):
            y.extend(["X"] * (len(y_truth) - len(y)))
        else:
            y_truth.extend(["X"] * (len(y) - len(y_truth)))
    return sum([1 for i in range(len(y)) if y[i] != y_truth[i]])

def compare_answers(thought: Thought, truth: str) -> float:
    """
    Calculate the percentage of errors in the answers.

    :param thought: The thought to evaluate
    :type thought: Thought
    :param truth: The truth (correct answers)
    :type truth: str
    :return: The percentage of errors in the answers, 0 means no error, 1 means all answers are wrong
    :rtype: float
    """
    y = thought.content.split(",")
    y_truth = truth.split(",")
    if len(y) != len(y_truth):
        if len(y) < len(y_truth):
            y.extend(["X"] * (len(y_truth) - len(y)))
        else:
            y_truth.extend(["X"] * (len(y) - len(y_truth)))
    return sum([1 for i in range(len(y)) if y[i] != y_truth[i]]) / len(y)

def io(task_input: dict, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the IO method.

    :param task_input: The input (passage, questions and options)
    :type task_input: dict
    :param task_truth: The truth (correct answers)
    :type task_truth: str
    :return: Challenge to run using the IO method
    :rtype: Challenge
    """
    root = Thought(json.dumps(task_input), [], [], is_executable=True)
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "reading_comprehension_prompt")
    )
    res = Evaluate(
        generate.get_children_thoughts(),
        lm,
        evaluate_function=compare_answers,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def cot(task_input: dict, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the COT method.

    :param task_input: The input (passage, questions and options)
    :type task_input: dict
    :param task_truth: The truth (correct answers)
    :type task_truth: str
    :return: Challenge to run using the COT method
    :rtype: Challenge
    """
    root = Thought(json.dumps(task_input), [], [], is_executable=True)
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "reading_comprehension_prompt_cot")
    )
    extract = Generate(
        generate.get_children_thoughts(),
        lm,
        1,
        generate_prompt=Parser.from_json(prompts, "extract_answers_prompt"),
    )
    res = Evaluate(
        extract.get_children_thoughts(),
        lm,
        evaluate_function=compare_answers,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def tot(task_input: dict, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the TOT method.

    :param task_input: The input (passage, questions and options)
    :type task_input: dict
    :param task_truth: The truth (correct answers)
    :type task_truth: str
    :return: Challenge to run using the TOT method
    :rtype: Challenge
    """
    root = Thought(json.dumps(task_input), [], [], is_executable=True)
    generate = Generate(
        [root], lm, 5, generate_prompt=Parser.from_json(prompts, "reading_comprehension_prompt")
    )
    scoring_layer = [
        Score([t], lm, scoring_function=num_errors, original_thought=root)
        for t in generate.get_children_thoughts()
    ]
    keep_best = KeepBestN([s.get_children_thoughts()[0] for s in scoring_layer], lm, 1, False)
    for _ in range(1):
        improve = Improve(
            keep_best.get_children_thoughts(),
            lm,
            5,
            original_thought=root,
            improve_prompt=Parser.from_json(prompts, "tot_improve_prompt"),
        )
        extract = [
            Generate(
                [g],
                lm,
                1,
                generate_prompt=Parser.from_json(prompts, "extract_answers_prompt"),
            )
            for g in improve.get_children_thoughts()
        ]
        scoring_layer = [
            Score(
                t.get_children_thoughts(), lm, scoring_function=num_errors, original_thought=root
            ).get_children_thoughts()[0]
            for t in extract
        ]
        keep_best = KeepBestN(scoring_layer, lm, 1, False)
    res = Evaluate(
        keep_best.get_children_thoughts(),
        lm,
        evaluate_function=compare_answers,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def got(task_input: dict, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the GOT method.

    :param task_input: The input (passage, questions and options)
    :type task_input: dict
    :param task_truth: The truth (correct answers)
    :type task_truth: str
    :return: Challenge to run using the GOT method
    :rtype: Challenge
    """
    root = Thought(json.dumps(task_input), [], [], is_executable=True)
    # Split questions into two groups
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "got_split_prompt")
    )
    split_questions = Split(generate.get_children_thoughts(), lm, ["Group 1", "Group 2"])
    keep_best_subgroup = []
    
    # Answer each group of questions
    for split_group in split_questions.get_children_thoughts():
        answer_group = Generate(
            [split_group],
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "reading_comprehension_prompt"),
        )
        # Improve answers
        improve = Improve(
            answer_group.get_children_thoughts(),
            lm,
            5,
            improve_prompt=Parser.from_json(prompts, "tot_improve_prompt"),
            original_thought=split_group,
        )
        extract = [
            Generate(
                [g],
                lm,
                1,
                generate_prompt=Parser.from_json(prompts, "extract_answers_prompt"),
            )
            for g in improve.get_children_thoughts()
        ]
        scoring_layer = [
            Score(
                t.get_children_thoughts(), lm, scoring_function=num_errors, original_thought=split_group
            ).get_children_thoughts()[0]
            for t in extract
        ]
        keep_best = KeepBestN(scoring_layer, lm, 1, False)
        keep_best_subgroup.append(keep_best.get_children_thoughts()[0])
    
    # Merge the answers
    aggregate_tries = Aggregate(
        keep_best_subgroup,
        lm,
        3,
        aggregate_prompt=Parser.from_json(prompts, "got_merge_prompt"),
    ).get_children_thoughts()
    scores = [
        Score([t], lm, scoring_function=num_errors, original_thought=root).get_children_thoughts()[0]
        for t in aggregate_tries
    ]
    keep_best_merged = KeepBestN(scores, lm, 1, False)
    res = Evaluate(
        keep_best_merged.get_children_thoughts(),
        lm,
        evaluate_function=compare_answers,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def fot(task_input: dict, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the FOT method.

    :param task_input: The input (passage, questions and options)
    :type task_input: dict
    :param task_truth: The truth (correct answers)
    :type task_truth: str
    :return: Challenge to run using the FOT method
    :rtype: Challenge
    """
    root = Thought("You want to generate 5 methods to answer reading comprehension questions.", [], [], is_executable=True)
    task_thought = Thought(json.dumps(task_input), [], [], is_executable=False)
    link_operation = Link([root], lm, [task_thought])
    generate_method = Generate(
        [root],
        lm,
        1,
        generate_prompt=Parser.from_json(prompts, "fot_generate_prompt"),
    )
    parse_methods = Split([generate_method.get_children_thoughts()[0]], lm, ["Method 1", "Method 2", "Method 3", "Method 4", "Method 5"])
    methods_thoughts = []
    
    for method_thought in parse_methods.get_children_thoughts():
        def concat_prompt(x:List[str])->List[str]:
            return [x[0]+x[1]]
            
        pits_gen_thought = Thought(f"Generate 3 pitfalls when doing the following process:{method_thought.content}", [], [], is_executable=True)
        link_operation = Link([method_thought], lm, [pits_gen_thought])
        pits_thoughts = Aggregate([pits_gen_thought,method_thought], lm, 1, aggregate_function=concat_prompt).get_children_thoughts()
        pits = Generate(
            pits_thoughts,
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "fot_check_prompt"),
        )
        parse_pits = Split([pits.get_children_thoughts()[0]], lm, ["Pitfall 1", "Pitfall 2", "Pitfall 3"])
        ego = Generate(
            [method_thought],
            lm,
            1,
            generate_prompt=Parser(prompt=f"Given the data {task_thought.content}, use the method {method_thought.content} to answer the questions.", name="use_method_prompt", plain=True),
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
            
        extract = Generate(
            ego.get_children_thoughts(),
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "extract_answers_prompt"),
        )
        methods_thoughts.append(extract.get_children_thoughts()[0])
        
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
        evaluate_function=compare_answers,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def reading_comprehension():
    """Main function for reading comprehension task"""
    global budget
    data_ids = [0]
    methods = [io, cot, tot, got, fot]
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "RACE_min")

    data = []
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r") as f:
                data.append([filename, json.load(f), ",".join(json.load(f)["answers"])])

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
