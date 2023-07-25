"""Tools for mocking OpenAI API responses (for testing purposes)."""


from __future__ import annotations  # noqa FI58


class MockCompletion:
    """Mock openai completion object."""

    def __init__(self, content: str | None = None, responses_per_request: int = 1):
        """Initialize a new instance of `MockCompletion` class.

        Args:
            content: The mocked content to be returned, i.e.,
                `json.dumps({"comment": "This is a great movie!",
                "label": 1})`.
            responses_per_request: Number of responses
                for each request.
        """
        # We generate 5 identical responses for each API call by default.
        if content is not None:
            # Mock an OpenAI ChatCompletion with identical responses.
            self.choices = [{"message": {"content": content}}] * responses_per_request
        else:
            # Mock an OpenAI ChatCompletion with different responses.
            # Only used in mock_batch_openai_response_with_different_completion.
            # The choice will be replaced later in the function.
            self.choices = []

    def __repr__(self):
        """Return a string representation.

        Returns:
            _string: A string representation of the object, including its choices.
        """
        _string = f"<MockObject choices={self.choices}>"
        return _string


def mock_one_openai_response(
    prompt: str,
    temperature: float,
    presence_penalty: float,
    frequency_penalty: float,
    content: str,
) -> MockCompletion:
    """Generate a mock completion object containing a choice with example content.

    This function creates a `MockCompletion`
    object with a `content` attribute set to an LLM completion string.

    Args:
        prompt: A mocked prompt that won't be used.
        temperature: A mocked temperature.
        presence_penalty: A mocked presence penalty.
        frequency_penalty: A mocked frequency penalty.
        content: The example string to be returned.

    Returns:
        A mock completion object simulating an OpenAI ChatCompletion API response.
    """
    _ = prompt, temperature, presence_penalty, frequency_penalty
    mock_completion = MockCompletion(content=content)
    return mock_completion


def mock_batch_openai_response_with_identical_completion(
    prompts: list[str],
    content: str,
    temperature: float,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    responses_per_request: int = 5,
    requests_per_minute: int = 80,
) -> list[MockCompletion]:
    """Generate a batch of  mock completion objects.

        This function creates a batch of `MockCompletion`
        object with a `content` attribute set to an LLM completion string.

    Args:
        prompts: A batch of mocked prompts that won't be used.
        content: The example string to be returned.
        temperature: A mocked temperature.
        presence_penalty: A mocked presence penalty.
        frequency_penalty: A mocked frequency penalty.
        responses_per_request: Number of responses for each request.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        A mock completion object simulating an OpenAI ChatCompletion API response.
    """
    _ = prompts, temperature, presence_penalty, frequency_penalty, requests_per_minute
    mock_completions = [
        MockCompletion(content=content, responses_per_request=responses_per_request)
        for _ in prompts
    ]
    return mock_completions


def mock_batch_openai_response_with_different_completion(
    prompts: list[str],
    content: str,
    temperature: float,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    responses_per_request: int = 5,
    requests_per_minute: int = 80,
) -> list[MockCompletion]:
    """Generate a batch of  mock completion objects.

        This function creates a batch of `MockCompletion`
        object with a `content` attribute set to an LLM completion string.

    Args:
        prompts: A batch of mocked prompts that won't be used.
        content: The example string to be returned.
        temperature: A mocked temperature.
        presence_penalty: A mocked presence penalty.
        frequency_penalty: A mocked frequency penalty.
        responses_per_request: Number of responses for each request.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        A mock completion object simulating an OpenAI ChatCompletion API response.
    """
    _ = prompts, temperature, presence_penalty, frequency_penalty, requests_per_minute
    mock_completions = [MockCompletion()] * 3
    mock_completions[0].choices = [
        {"message": {"content": '{"input": "1", "output": "a"}'}},
        {"message": {"content": '{"input": "1", "output": "b"}'}},
        {"message": {"content": '{"input": "1", "output": "a"}'}},
        {"message": {"content": '{"input": "1", "output": "c"}'}},
        {"message": {"content": '{"input": "2", "output": "a"}'}},
        {"message": {"content": '{"input": "2", "output": "b"}'}},
        {"message": {"content": '{"input": "3", "output": "a"}'}},
        {"message": {"content": '{"input": "3", "output": "b"}'}},
        {"message": {"content": '{"input": "3", "output": "a"}'}},
    ]
    mock_completions[1].choices = [
        {"message": {"content": '{"input": "1", "output": "a"}'}},
        {"message": {"content": '{"input": "1", "output": "b"}'}},
        {"message": {"content": '{"input": "1", "output": "a"}'}},
        {"message": {"content": '{"input": "1", "output": "c"}'}},
        {"message": {"content": '{"input": "2", "output": "a"}'}},
        {"message": {"content": '{"input": "2", "output": "b"}'}},
        {"message": {"content": '{"input": "3", "output": "a"}'}},
        {"message": {"content": '{"input": "3", "output": "b"}'}},
        {"message": {"content": '{"input": "3", "output": "a"}'}},
    ]
    mock_completions = [
        MockCompletion(
            content='{"input": "This is a great movie!", "output": "1"}',
            responses_per_request=1,
        )
        for _ in prompts
    ]
    return mock_completions
