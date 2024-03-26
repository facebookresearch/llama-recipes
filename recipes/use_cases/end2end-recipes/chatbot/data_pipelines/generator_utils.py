# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import re
from transformers import  AutoTokenizer
from octoai.client import Client
import asyncio
import magic
from PyPDF2 import PdfReader
import json
from doc_processor import split_text_into_chunks
import logging
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip() + ' '
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
    return ''

def read_pdf_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            num_pages = len(pdf_reader.pages)
            file_text = [pdf_reader.pages[page_num].extract_text().strip() + ' ' for page_num in range(num_pages)]
            return ''.join(file_text)
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
    return ''

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Assuming each item in the list has a 'question' and 'answer' key
            # Concatenating question and answer pairs with a space in between and accumulating them into a single string
            file_text = ' '.join([item['question'].strip() + ' ' + item['answer'].strip() + ' ' for item in data])
            return file_text
    except Exception as e:
        logging.error(f"Error reading JSON file {file_path}: {e}")
    return ''


def process_file(file_path):
    file_type = magic.from_file(file_path, mime=True)
    if file_type in ['text/plain', 'text/markdown', 'JSON']:
        return read_text_file(file_path)
    elif file_type == 'application/pdf':
        return read_pdf_file(file_path)
    else:
        logging.warning(f"Unsupported file type {file_type} for file {file_path}")
        return ''

def read_file_content(context):
    file_strings = []

    for root, _, files in os.walk(context['data_dir']):
        for file in files:
            file_path = os.path.join(root, file)
            file_text = process_file(file_path)
            if file_text:
                file_strings.append(file_text)

    return ' '.join(file_strings)



def parse_qa_to_json(response_string):
    # Adjusted regex to capture question-answer pairs more flexibly
    # This pattern accounts for optional numbering and different question/answer lead-ins
    pattern = re.compile(
        r"\d*\.\s*Question:\s*(.*?)\nAnswer:\s*(.*?)(?=\n\d*\.\s*Question:|\Z)", 
        re.DOTALL
    )

    # Find all matches in the response string
    matches = pattern.findall(response_string)

    # Convert matches to a structured format
    qa_list = [{"question": match[0].strip(), "answer": match[1].strip()} for match in matches]

    # Convert the list to a JSON string
    return json.dumps(qa_list, indent=4)


async def prepare_and_send_request(chat_service, api_context: dict, document_content: str, total_questions: int) -> dict:
    prompt_for_system = api_context['question_prompt_template'].format(total_questions=total_questions, language=api_context["language"])
    chat_request_payload = [{'role': 'system', 'content': prompt_for_system}, {'role': 'user', 'content': document_content}]
    return json.loads(await chat_service.execute_chat_request_async(api_context, chat_request_payload))

async def generate_question_batches(chat_service, api_context: dict):
    document_text = read_file_content(api_context)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
    document_batches = split_text_into_chunks(api_context, document_text, tokenizer)
    
    total_questions = api_context["total_questions"]
    batches_count = len(document_batches)
    base_questions_per_batch = total_questions // batches_count
    extra_questions = total_questions % batches_count

    print(f"Questions per batch: {base_questions_per_batch} (+1 for the first {extra_questions} batches), Total questions: {total_questions}, Batches: {batches_count}")
    generation_tasks = []
    for batch_index, batch_content in enumerate(document_batches):
        print(f"len of batch_content: {len(batch_content)}, batch_index: {batch_index}")
        #Distribute extra questions across the first few batches
        questions_in_current_batch = base_questions_per_batch + (1 if batch_index < extra_questions else 0)
        print(f"Batch {batch_index + 1} - {questions_in_current_batch} questions ********")
        generation_tasks.append(prepare_and_send_request(chat_service, api_context, batch_content, questions_in_current_batch))

    question_generation_results = await asyncio.gather(*generation_tasks)

    return question_generation_results



