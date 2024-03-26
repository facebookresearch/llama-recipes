# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Assuming result_average_token is a constant, use UPPER_CASE for its name to follow Python conventions
AVERAGE_TOKENS_PER_RESULT = 100

def get_token_limit_for_model(model: str) -> int:
    """Returns the token limit for a given model."""
    if model == "llama-2-70b-chat-fp16" or model == "llama-2-13b-chat-turbo":
        return 4096
    

def calculate_num_tokens_for_message(encoded_text) -> int:
    """Calculates the number of tokens used by a message."""
    
    # Added 3 to account for priming with assistant's reply, as per original comment
    return len(encoded_text) + 3


def split_text_into_chunks(context: dict, text: str, tokenizer) -> list[str]:
    """Splits a long text into substrings based on token length constraints, adjusted for question generation."""
    # Adjusted approach to calculate max tokens available for text chunks
    encoded_text = tokenizer(text, return_tensors="pt", padding=True)["input_ids"]
    encoded_text = encoded_text.squeeze()
    model_token_limit = get_token_limit_for_model(context["model"])

    tokens_for_questions = calculate_num_tokens_for_message(encoded_text)
    estimated_tokens_per_question = AVERAGE_TOKENS_PER_RESULT
    estimated_total_question_tokens = estimated_tokens_per_question * context["total_questions"]
    # Ensure there's a reasonable minimum chunk size
    max_tokens_for_text = max(model_token_limit - tokens_for_questions - estimated_total_question_tokens, model_token_limit // 10)
    
    chunks, current_chunk = [], []
    print(f"Splitting text into chunks of {max_tokens_for_text} tokens, encoded_text {len(encoded_text)}", flush=True)
    for token in encoded_text:
        if len(current_chunk) >= max_tokens_for_text:
            chunks.append(tokenizer.decode(current_chunk).strip())
            current_chunk = []
        else:
            current_chunk.append(token)

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk).strip())

    print(f"Number of chunks in the processed text: {len(chunks)}", flush=True)
   
    return chunks