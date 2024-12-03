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
with open(os.path.join(os.path.dirname(__file__), "sort_prompt.json"), "r") as f:
    prompts = json.load(f)

def str_to_list(s: str) -> list:
    return [int(x) for x in s[1:-1].split(",")]


def num_errors(thought: Thought, original_thought: Thought) -> int:
    """
    Calculate the number of errors in the sorted array.

    :param thought: The thought to evaluate
    :type thought: Thought
    :return: The number of errors in the sorted array
    :rtype: int
    """
    y = str_to_list(thought.content)
    y_truth = sorted(str_to_list(original_thought.content))
    if len(y) != len(y_truth):
        if len(y) < len(y_truth):
            y.extend([-1] * (len(y_truth) - len(y)))
        else:
            y_truth.extend([-1] * (len(y) - len(y_truth)))
    return sum([1 for i in range(len(y)) if y[i] != y_truth[i]])+abs(len(y)-len(y_truth))


def compare_sorted(thought: Thought, truth: str) -> float:
    """
    Calculate the percentage of errors in the sorted array.

    :param thought: The thought to evaluate
    :type thought: Thought
    :param truth: The truth (sorted array)
    :type truth: str
    :return: The percentage of errors in the sorted array, 0 means no error, 1 means all elements are different
    :rtype: float
    """
    y = str_to_list(thought.content)
    y_truth = sorted(str_to_list(truth))
    # filling the shorter array with -1
    if len(y) != len(y_truth):
        if len(y) < len(y_truth):
            y.extend([-1] * (len(y_truth) - len(y)))
        else:
            y_truth.extend([-1] * (len(y) - len(y_truth)))
    return sum([1 for i in range(len(y)) if y[i] != y_truth[i]]) / len(y)


def io(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the IO method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the IO method
    :rtype: Challenge
    """
    root = Thought(task_input, [], [], is_executable=True)
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "sort_prompt")
    )
    res = Evaluate(
        generate.get_children_thoughts(),
        lm,
        evaluate_function=compare_sorted,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)


def cot(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the COT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the COT method
    :rtype: Challenge
    """
    root = Thought(task_input, [], [], is_executable=True)
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "sort_prompt_cot")
    )
    extract = Generate(
        generate.get_children_thoughts(),
        lm,
        1,
        generate_prompt=Parser.from_json(prompts, "extract_sorted_list_prompt"),
    )
    res = Evaluate(
        extract.get_children_thoughts(),
        lm,
        evaluate_function=compare_sorted,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)


def tot(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the TOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the TOT method
    :rtype: Challenge
    """
    root = Thought(task_input, [], [], is_executable=True)

    generate = Generate(
        [root], lm, 5, generate_prompt=Parser.from_json(prompts, "sort_prompt")
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
                generate_prompt=Parser.from_json(prompts, "extract_sorted_list_prompt"),
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
        evaluate_function=compare_sorted,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)

def got(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the GOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the GOT method
    :rtype: Challenge
    """
    root = Thought(task_input, [], [], is_executable=True)
    generate = Generate(
        [root], lm, 1, generate_prompt=Parser.from_json(prompts, "got_split_prompt")
    )
    split_lists = Split(generate.get_children_thoughts(), lm, ["List 1", "List 2"])
    keep_best_sublist = []
    # sort each sublist
    for split_list in split_lists.get_children_thoughts():
        sort_split_list = Generate(
            [split_list],
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "sort_prompt"),
        )
        # single layer improvement
        improve = Improve(
            sort_split_list.get_children_thoughts(),
            lm,
            5,
            improve_prompt=Parser.from_json(prompts, "tot_improve_prompt"),
            original_thought=split_list,
        )
        extract = [
            Generate(
                [g],
                lm,
                1,
                generate_prompt=Parser.from_json(prompts, "extract_sorted_list_prompt"),
            )
            for g in improve.get_children_thoughts()
        ]
        scoring_layer = [
            Score(
                t.get_children_thoughts(), lm, scoring_function=num_errors, original_thought=split_list
            ).get_children_thoughts()[0]
            for t in extract
        ]
        keep_best = KeepBestN(scoring_layer, lm, 1, False)
        keep_best_sublist.append(keep_best.get_children_thoughts()[0])
    # merge the sorted sublists with 3 tries
    aggregate_tries = Aggregate(
        keep_best_sublist,
        lm,
        3,
        aggregate_prompt=Parser.from_json(prompts, "got_merge_prompt"),
    ).get_children_thoughts()
    scores = [
        Score([t], lm, scoring_function=num_errors, original_thought=root).get_children_thoughts()[0]
        for t in aggregate_tries
    ]
    # evaluate the merged lists
    keep_best_merged = KeepBestN(scores, lm, 1, False)
    res = Evaluate(
        keep_best_merged.get_children_thoughts(),
        lm,
        evaluate_function=compare_sorted,
        ground_truth=task_truth,
    )
    return Challenge(root, max_budget=budget)


def fot(task_input: str, task_truth: str) -> Challenge:
    """
    Generates the Challenge to run using the FOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the FOT method
    :rtype: Challenge
    """
    # brainstorming
    root = Thought("You want to generate 5 methods to sort a list in ascending order.", [], [], is_executable=True)
    task_thought = Thought(task_input, [], [], is_executable=False)
    link_operation = Link([root], lm, [task_thought])
    generate_method = Generate(
        [root],
        lm,
        1,
        generate_prompt=Parser.from_json(prompts, "fot_generate_prompt"),
    )
    parse_methods = Split([generate_method.get_children_thoughts()[0]], lm, ["Method 1", "Method 2", "Method 3", "Method 4", "Method 5"])
    methods_thoughts = []
    # generate results by each method
    for method_thought in parse_methods.get_children_thoughts():
        def concat_prompt(x:List[str])->List[str]:
            return [x[0]+x[1]]
        # try methods
        pits_gen_thought = Thought(f"Generate 3 pitfalls when doing the following process:{method_thought.content}", [], [], is_executable=True)
        # link thought
        link_operation = Link([method_thought], lm, [pits_gen_thought])
        pits_thoughts = Aggregate([pits_gen_thought,method_thought], lm, 1, aggregate_function=concat_prompt).get_children_thoughts()
        pits = Generate(
            pits_thoughts,
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "fot_check_prompt"),
        )
        parse_pits = Split([pits.get_children_thoughts()[0]], lm, ["Pitfall 1", "Pitfall 2", "Pitfall 3"])
        ego=Generate(
            [method_thought],
            lm,
            1,
            generate_prompt=Parser(prompt=f"Given the data {task_thought.content}, use the method {method_thought.content} to sort the data in ascending order.", name="use_method_prompt", plain=True),
        )
        # improve the result
        for pitfall in parse_pits.get_children_thoughts():
            improve_parser=Parser.from_json(prompts, "fot_improve_prompt")
            improve_parser.cot=pitfall.content
            # create link operation
            link_operation = Link([pitfall], lm, [ego.get_children_thoughts()[0]])
            ego = Improve(
                link_operation.get_children_thoughts(),
                lm,
                1,
                improve_prompt=improve_parser,
                original_thought=task_thought,
            )
        # finalize result
        extract = Generate(
            ego.get_children_thoughts(),
            lm,
            1,
            generate_prompt=Parser.from_json(prompts, "extract_sorted_list_prompt"),
        )
        methods_thoughts.append(extract.get_children_thoughts()[0])
    # evaluate the results
    final_scores = []
    for method in methods_thoughts:
        final_scores.append(
            Score(
                [method],
                lm,
                scoring_function=num_errors,
                original_thought=task_thought
            ).get_children_thoughts()[0]
        )
    # keep the best result
    keep_best = KeepBestN(final_scores, lm, 1, False)
    res = Evaluate(keep_best.get_children_thoughts(), lm, evaluate_function=compare_sorted, ground_truth=task_truth)
    return Challenge(root, max_budget=budget)


# main function for sorting_032
def sorting_032():
    global budget
    # this section is for testing the sorting task
    # data_ids = [item for item in range(0, 100)]
    data_ids = [0]
    methods = [io,cot,tot,got,fot]
    # methods = [tot]
    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "sorting_032.csv")

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