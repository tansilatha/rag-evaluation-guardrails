import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LLAMA_GUARD_MODEL = "hf.co/QuantFactory/Meta-Llama-Guard-2-8B-GGUF:Q4_K_M"
LLAMA_GUARD_URL = "http://localhost:11434/api/chat"
INPUT_GUARD_INSTRUCTION = (
    "Task: Check if there is unsafe content in the user messages. "
    "You are a strict safety evaluator. You must respond with exactly one "
    "of these categories: 'safe' or 'unsafe'. If the message is unsafe, "
    "also state which category it violates."
)
OUTPUT_GUARD_INSTRUCTION = (
    "Task: Check if the assistant response contains unsafe content. "
    "You are a strict safety evaluator. You must respond with exactly one "
    "of these categories: 'safe' or 'unsafe'. If the response is unsafe, "
    "also state which category it violates."
)


def _call_ollama(prompt: str, model: str, url: str, timeout: int = 120) -> str:
    """Send a single user message to an Ollama model and return the text reply.

    Raises a RuntimeError on any connection, HTTP, or response-format problem.
    """
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {details}"
        ) from error
    except URLError as error:
        raise ConnectionError(
            "Could not connect to Ollama. Make sure Ollama is running at "
            f"{url}."
        ) from error
    except TimeoutError as error:
        raise ConnectionError("The request to Ollama timed out.") from error

    if not isinstance(result, dict):
        raise ValueError("Ollama returned an unexpected response.")
    message = result.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Ollama response did not contain a message with text.")
    return message["content"]


def check_safety(text: str, guard_type: str = "input") -> tuple[bool, str]:
    """Run Llama Guard 2 over text, blocking on failures or unexpected labels."""
    if guard_type not in {"input", "output"}:
        raise ValueError("guard_type must be 'input' or 'output'.")
    instruction = (
        INPUT_GUARD_INSTRUCTION
        if guard_type == "input"
        else OUTPUT_GUARD_INSTRUCTION
    )
    message_label = "User message" if guard_type == "input" else "Assistant response"
    prompt = f"{instruction}\n\n{message_label}:\n{text}"
    try:
        reply = _call_ollama(prompt, LLAMA_GUARD_MODEL, LLAMA_GUARD_URL)
    except (ConnectionError, RuntimeError, ValueError) as error:
        return False, (
            f"Could not run the {guard_type} safety check "
            f"(Llama Guard request failed: {error}). The request is blocked "
            "for safety."
        )

    normalized = reply.strip().casefold()
    if not normalized:
        return False, (
            f"The {guard_type} safety check returned an empty response. "
            "The request is blocked for safety."
        )
    first_token = normalized.split()[0].strip(".,;:!?\"'`")

    if first_token == "safe":
        return True, f"The {guard_type} is safe."
    if first_token == "unsafe":
        return False, (
            f"The {guard_type} was blocked by Llama Guard 2 as unsafe. "
            f"Guard response: {reply.strip()}"
        )
    return False, (
        f"The {guard_type} safety check returned an unexpected response "
        f"({reply.strip()!r}). The request is blocked for safety."
    )
