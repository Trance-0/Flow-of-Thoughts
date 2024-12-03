import os
from anthropic import Anthropic
from typing import List, Dict, Union, Any
from .abstract_language_model import AbstractLanguageModel


class Claude(AbstractLanguageModel):
    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False
    ) -> None:
        """Initialize Claude API client"""
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for chatgpt, i.e. gpt-4, gpt-3.5-turbo, etc.
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1M prompt tokens and 1M response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at (it will not generate the stop sequence).
        self.stop: Union[str, List[str]] = self.config["stop"]
        self.api_key: str = os.getenv("ANTHROPIC_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("ANTHROPIC_API_KEY is not set")
        # Initialize the OpenAI Client
        self.client = Anthropic(api_key=self.api_key)
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Query Claude API and return response(s)

        :param query: Query string to send to Claude
        :param num_responses: Number of responses to generate (currently Claude API only supports 1)
        :return: Claude API response object
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": query}]
            )

            if self.cache:
                self.response_cache[query] = response

            self.prompt_tokens += response.usage.input_tokens
            self.completion_tokens += response.usage.output_tokens
            self.cost += self.get_response_cost(response)

            return response

        except Exception as e:
            self.logger.error(f"Error querying Claude API: {str(e)}")
            raise

    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        Extract text from Claude API response(s)

        :param query_responses: Claude API response object(s)
        :return: List of response text strings
        """
        if isinstance(query_responses, list):
            return [response.content[0].text for response in query_responses]
        return [query_responses.content[0].text]

    def get_response_cost(self, query_response: Union[List[Any], Any]) -> float:
        """
        Calculate cost of Claude API response

        :param query_response: Claude API response object
        :return: Cost in USD
        """
        # Claude API pricing per 1K tokens (as of 2024)
        input_cost = 0.008  # USD per 1K input tokens
        output_cost = 0.024  # USD per 1K output tokens

        if isinstance(query_response, list):
            total_cost = 0.0
            for response in query_response:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_cost += (input_tokens * input_cost + output_tokens * output_cost) / 1000
            return total_cost

        input_tokens = query_response.usage.input_tokens
        output_tokens = query_response.usage.output_tokens
        return (input_tokens * input_cost + output_tokens * output_cost) / 1000
