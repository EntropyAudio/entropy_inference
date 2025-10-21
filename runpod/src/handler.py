import runpod
from converter import extract_input
from controller import run_inference
from utils.utils import setup_logger
import utils.constants as c

model = None

def handler(event):
    logger = setup_logger()
    logger.info(f"Received event: {event}")
    input = extract_input(event)
    result = run_inference(input, model)
    return result

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
