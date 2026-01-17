from .utils import constants as c
from .utils.exceptions import InvalidPromptError, InvalidRequestError
import logging

logger = logging.getLogger(c.LOGGER_NAME)

def extract_input(event):
    """
    Extracts fields needed for inference from the input event.

    Args:
        event: The input event.

    Returns:
        The batch size and conditioning.
    """
    logger.info(f"Extracting and validating input...")
    if "input" not in event:
        raise InvalidRequestError(request=event)

    input = event.get("input")
    prompt = input.get("prompt")
    batch_size = input.get("batch_size")

    if not prompt or not isinstance(prompt, str) or prompt == "":
        raise InvalidPromptError(prompt=prompt)

    return prompt, batch_size
