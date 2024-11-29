import csv
import datetime
import json
import logging
import os
from language_models.llamachat_hf import LlamaHF
from language_models.chatgpt import ChatGPT
from thoughts.thought import Thought
from thoughts.operations import Generate, Evaluate, KeepBestN, Parser
from thoughts.challenge import Challenge
budget = 30

# use universal language model
lm_name = "chatgpt4o-mini"
lm = ChatGPT(
    model_name=lm_name,
    cache=True,
)

# lm_name = "llama3.2-1b-instruct-hf"
# lm = LlamaHF(
#     model_name=lm_name,
#     cache=True,
# )

prompts=None

# load prompts
with open(os.path.join(os.path.dirname(__file__), "prompt.json"), "r") as f:
    prompts = json.load(f)

def num_errors(thought: Thought)->float:
    y=list(thought.content)
    y_truth=sorted(y)
    return sum([1 for i in range(len(y)) if y[i]!=y_truth[i]])

def compare_sorted(thought: Thought, truth: str)->float:
    y=list(thought.content)
    y_truth=list(truth)
    return sum([1 for i in range(len(y)) if y[i]!=y_truth[i]])/len(y)

def io(task_input: str, task_truth: str)->Challenge:
    """
    Generates the Challenge to run using the IO method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the IO method
    :rtype: Challenge
    """
    root = Thought(task_input, [],[],is_executable=True)
    generate = Generate([root], lm, 1,Parser.from_dict(prompts["sort_prompt"]))
    res=Evaluate(generate.get_children_thoughts(), lm, evaluate_function=num_errors, ground_truth=task_truth)
    return Challenge(root,max_budget=budget)

def cot(task_input: str, task_truth: str)->Challenge:
    """
    Generates the Challenge to run using the COT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the COT method
    :rtype: Challenge
    """
    root = Thought(task_input,[],[], is_executable=True)
    generate = Generate([root], lm, 1,Parser.from_dict(prompts["sort_prompt_cot"]))
    res=Evaluate(generate.get_children_thoughts(), lm, evaluate_function=compare_sorted, ground_truth=task_truth)
    return Challenge(root,max_budget=budget)

def tot(task_input: str, task_truth: str)->Challenge:
    """
    Generates the Challenge to run using the TOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the TOT method
    :rtype: Challenge
    """
    root = Thought(task_input,[],[], is_executable=True)

    generate = Generate([root], lm, 20, Parser.from_dict(prompts["sort_prompt_tot"]))
    score = Evaluate(generate.get_children_thoughts(), lm, evaluate_function=num_errors, ground_truth=task_truth)
    keep_best = KeepBestN(score.get_children_thoughts(),1, False)
    for _ in range(1):
        generate = Generate(keep_best.get_children_thoughts(), lm, 20, Parser.from_dict(prompts["sort_prompt_tot"]))
        score = Evaluate(generate.get_children_thoughts(), lm, evaluate_function=num_errors, ground_truth=task_truth)
        keep_best = KeepBestN(score.get_children_thoughts(),1, False)
    res=Evaluate(keep_best.get_children_thoughts(), lm, evaluate_function=compare_sorted, ground_truth=task_truth)
    return Challenge(root,max_budget=budget)


def tot2(task_input: str, task_truth: str)->Challenge:
    """
    Generates the Challenge to run using the TOT2 method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the TOT2 method
    :rtype: Challenge
    """
    root = Thought(task_input,[],[], is_executable=True)
    generate = Generate([root], lm, 20, Parser.from_dict(prompts["sort_prompt_tot"]))
    score = Evaluate(generate.get_children_thoughts(), lm, evaluate_function=num_errors, ground_truth=task_truth)
    keep_best=KeepBestN(score.get_children_thoughts(),1, False)
    for _ in range(2):
        generate = Generate(keep_best.get_children_thoughts(), lm, 20, Parser.from_dict(prompts["sort_prompt_tot"]))
        score = Evaluate(generate.get_children_thoughts(), lm, evaluate_function=num_errors, ground_truth=task_truth)
        keep_best = KeepBestN(score.get_children_thoughts(),1, False)
    res=Evaluate(keep_best.get_children_thoughts(), lm, evaluate_function=compare_sorted, ground_truth=task_truth)
    return Challenge(root,max_budget=budget)

def got(task_input: str, task_truth: str)->Challenge:
    """
    Generates the Challenge to run using the GOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the GOT method
    :rtype: Challenge
    """
    root = Thought(task_input,[],[], is_executable=True)
    generate = Generate([root], lm, 20, Parser.from_dict(prompts["got_split_prompt"]))
    for i in range(3):
        score = Evaluate(generate.get_children_thoughts(), lm, evaluate_function=num_errors, ground_truth=task_truth)
        keep_best = KeepBestN(score.get_children_thoughts(),1, False)
        generate = Generate(keep_best.get_children_thoughts(), lm, 20, Parser.from_dict(prompts["got_merge_prompt"]))
    res=Evaluate(generate.get_children_thoughts(), lm, evaluate_function=compare_sorted, ground_truth=task_truth)
    return Challenge(root,max_budget=budget)

def fot(task_input: str, task_truth: str)->Challenge:
    """
    Generates the Challenge to run using the FOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the FOT method
    :rtype: Challenge
    """
    root = Thought(0, "root", {})
    return Challenge(root)

# main function for sorting_032
def sorting_032():
    global budget
    # this section is for testing the sorting task
    data_ids = [item for item in range(0, 100)]
    methods = [io, cot, tot, tot2]
    # methods = [io,cot,tot,tot2,got,fot]
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
            cur_challenge = method(data[1],data[2])
            try:
                cur_challenge.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}.json",
            )
            res = cur_challenge.__json__()
            with open(path, "w") as f:
                json.dump(res, f)
            budget = cur_challenge.get_remaining_budget()

    logging.info(f"budget remains {budget}")
