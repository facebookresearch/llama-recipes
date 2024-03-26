# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse
import asyncio
import json
from config import load_config
from generator_utils import generate_question_batches, parse_qa_to_json
from itertools import chain
import logging
import aiofiles  # Ensure aiofiles is installed for async file operations
from abc import ABC, abstractmethod
from octoai.client import Client
from functools import partial

# Configure logging to include the timestamp, log level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Manage rate limits with throttling
rate_limit_threshold = 2000
allowed_concurrent_requests = int(rate_limit_threshold * 0.75)
request_limiter = asyncio.Semaphore(allowed_concurrent_requests)

class ChatService(ABC):
    @abstractmethod
    async def execute_chat_request_async(self, api_context: dict, chat_request):
        pass

# Please implement your own chat service class here.
# The class should inherit from the ChatService class and implement the execute_chat_request_async method.
class OctoAIChatService(ChatService):
    async def execute_chat_request_async(self, api_context: dict, chat_request):
        async with request_limiter:
            try:
                event_loop = asyncio.get_running_loop()
                client = Client(api_context['api_key'])
                api_chat_call = partial(
                    client.chat.completions.create,
                    model=api_context['model'],
                    messages=chat_request,
                    temperature=0.0
                )
                response = await event_loop.run_in_executor(None, api_chat_call)
                assistant_response = next((choice.message.content for choice in response.choices if choice.message.role == 'assistant'), "")
                assistant_response_json = parse_qa_to_json(assistant_response)
                      
                return assistant_response_json
            except Exception as error:
                print(f"Error during chat request execution: {error}")
                return ""
            
async def main(context):
    chat_service = OctoAIChatService()
    try:
        logging.info("Starting to generate question/answer pairs.")
        data = await generate_question_batches(chat_service, context)
        if not data:
            logging.warning("No data generated. Please check the input context or model configuration.")
            return
        flattened_list = list(chain.from_iterable(data))
        logging.info(f"Successfully generated {len(flattened_list)} question/answer pairs.")
        # Use asynchronous file operation for writing to the file
        async with aiofiles.open("data.json", "w") as output_file:
            await output_file.write(json.dumps(flattened_list, indent=4))
        logging.info("Data successfully written to 'data.json'. Process completed.")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the process: {e}")

def parse_arguments():
    # Define command line arguments for the script
    parser = argparse.ArgumentParser(
        description="Generate question/answer pairs from documentation."
    )
    parser.add_argument(
        "-t", "--total_questions",
        type=int,
        default=10,
        help="Specify the number of question/answer pairs to generate."
    )
    parser.add_argument(
        "-m", "--model",
        choices=["llama-2-70b-chat-fp16", "llama-2-13b-chat-fp16"],
        default="llama-2-70b-chat-fp16",
        help="Select the model to use for generation."
    )
    parser.add_argument(
        "-c", "--config_path",
        default="config.yaml",
        help="Set the configuration file path that has system prompt along with language, dataset path and number of questions."
    )
    return parser.parse_args()

if __name__ == "__main__":
    logging.info("Initializing the process and loading configuration...")
    args = parse_arguments()

    context = load_config(args.config_path)
    context["total_questions"] = args.total_questions
    context["model"] = args.model

    logging.info(f"Configuration loaded. Generating {args.total_questions} question/answer pairs using model '{args.model}'.")
    asyncio.run(main(context))