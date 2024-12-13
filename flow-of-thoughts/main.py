from test.check_models import check_models, test_generate_operation
from language_models.chatgpt import ChatGPT
from language_models.llamachat_hf import LlamaHF
from test.sorting_test.sorting_test import sorting_032
from test.set_intersections.set_intersection_test import set_intersection_064
# from test.reading_comprehension.reading_comprehension import reading_comprehension
import logging
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_logger(task_name: str):
    log_dir = os.path.join(BASE_DIR, "logs")
    log_file = os.path.join(log_dir, f"{task_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
        level=logging.DEBUG,
    )
    logging.getLogger().addHandler(logging.StreamHandler())


if __name__ == "__main__":
    # check_models(ChatGPT(model_name="chatgpt4o"))
    # check_models(LlamaHF(model_name="llama3.2-1b-hf"))

    # print("\nTesting Generate operation...")
    # test_generate_operation(ChatGPT(model_name="chatgpt4o"))

    # setup logger for all tasks
    # setup_logger("sorting_032")
    # sorting_032()
    setup_logger("set_intersection_064")
    set_intersection_064()
    # setup_logger("reading_comprehension_064")
    # reading_comprehension()
