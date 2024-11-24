import csv
import datetime
import json
import logging
import os
import operations
import language_models
import thoughts
budget = 30

# use universal language model
lm_name = "chatgpt4o"
lm = language_models.ChatGPT(
    model_name=lm_name,
    cache=True,
)

# load prompts
with open(os.path.join(os.path.dirname(__file__), "prompt.json"), "r") as f:
    prompts = json.load(f)

def io(task_input: str, task_truth: str)->thoughts.Challenge:
    """
    Generates the Challenge to run using the IO method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the IO method
    :rtype: Challenge
    """
    root = thoughts.Thought(task_input, is_executable=True)

def cot(task_input: str, task_truth: str)->thoughts.Challenge:
    """
    Generates the Challenge to run using the COT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the COT method
    :rtype: Challenge
    """
    root = thoughts.Thought(0, "root", {})
    return root

def tot(task_input: str, task_truth: str)->thoughts.Challenge:
    """
    Generates the Challenge to run using the TOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the TOT method
    :rtype: Challenge
    """
    root = thoughts.Thought(0, "root", {})
    return root

def tot2(task_input: str, task_truth: str)->thoughts.Challenge:
    """
    Generates the Challenge to run using the TOT2 method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the TOT2 method
    :rtype: Challenge
    """
    root = thoughts.Thought(0, "root", {})
    return root

def got(task_input: str, task_truth: str)->thoughts.Challenge:
    """
    Generates the Challenge to run using the GOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the GOT method
    :rtype: Challenge
    """
    root = thoughts.Thought(0, "root", {})
    return root

def fot(task_input: str, task_truth: str)->thoughts.Challenge:
    """
    Generates the Challenge to run using the FOT method.

    :param task_input: The input (unsorted array)
    :type task_input: str
    :param task_truth: The truth (sorted array)
    :type task_truth: str
    :return: Challenge to run using the FOT method
    :rtype: Challenge
    """
    root = thoughts.Thought(0, "root", {})
    return root

data_ids = [item for item in range(0, 100)]
methods = [io, cot, tot, tot2, got, fot]
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

logging.basicConfig(
    filename=os.path.join(results_folder, "log.log"),
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

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
        lm = language_models.ChatGPT(
            os.path.join(
                os.path.dirname(__file__),
                "../../graph-of-thoughts/graph_of_thoughts/language_models/config.json",
            ),
            model_name=lm_name,
            cache=True,
        )
        try:
            method(data[1],data[2]).run()
        except Exception as e:
            logging.error(f"Exception: {e}")
        path = os.path.join(
            results_folder,
            method.__name__,
            f"{data[0]}.json",
        )
        res = method().output_graph(path)
        budget -= res.cost

logging.info(f"budget remains {orig_budget-budget}")