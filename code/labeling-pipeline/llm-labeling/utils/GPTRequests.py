import os
import json
from openai import OpenAI
from dotenv import dotenv_values
from collections import defaultdict
import typing_extensions as typing
import tiktoken

"""
secrets = dotenv_values(".env")
api_key = secrets['OPENAI_KEY']

client = OpenAI(
    api_key=api_key    
)"""

def get_response(client, model_name, prompt, temperature):
    ct, num_tokens, num_completion_tokens, num_prompt_tokens = 0, 0, 0, 0
    response = ""
    
    failed = False
    while ct < 3:
        ct += 1
        try:
            response = client.chat.completions.create(
                model=model_name, # either gpt-4 or gpt-4o
                messages=prompt, 
                temperature=temperature,  # Control the randomness of the output
                response_format={"type": "json_object"}  # Ensure output is in JSON format
            )
            num_tokens += response.usage.total_tokens
            num_completion_tokens += response.usage.completion_tokens
            num_prompt_tokens += response.usage.prompt_tokens
            
            # checking well-formed json content
            output = response.choices[0].message.content
            try:
                json_output = json.loads(output)
                
                if failed:
                    print("Succeeded Retry. Continuing")
                
                return response, num_tokens, num_completion_tokens, num_prompt_tokens #.choices[0].message.content
            except json.JSONDecodeError as json_error:
                print(f"Malformed JSON content: {json_error}. Retrying ({ct}/3)...")
                failed = True
                continue  # Retry the request
        except Exception as e:
            print("Error")
            print(e)
            print(prompt)
            continue
            #logging.error(traceback.format_exc())
    return response, num_tokens, num_completion_tokens, num_prompt_tokens

"""
Helper function to iteratively evaluate prompts through the LLM (e.g., model_name) at the given temperature.

Returns the token usage and dictionary (video id to output mapping)
"""
def evaluate_prompts(client, provided_prompt, model_name, temperature):
    vid_to_output = defaultdict()

    # intermediate variables
    total_completion_token = 0
    total_prompt_token = 0

    # iterating through each prompt into the LLM
    for i, (vid, prompt) in enumerate(provided_prompt.items()):
        if i % 30 ==0:
            print(i)
        
        response, num_tokens, num_completion_tokens, num_prompt_tokens = get_response(client, model_name, prompt, temperature)

        # saving the outputs
        total_completion_token += num_completion_tokens
        total_prompt_token += num_prompt_tokens
        vid_to_output[vid] = response

    print("Total Number of Input Token: " + str(total_prompt_token))
    print("Total Number of Output Token: " + str(total_completion_token))
    return vid_to_output, total_completion_token, total_prompt_token


def get_response_gemini(client, model_name, prompt, temperature):
    ct, num_tokens, num_completion_tokens, num_prompt_tokens = 0, 0, 0, 0
    response = ""
    while ct < 3:
        ct += 1
        try:
            response = client.generate_content(
                prompt, 
                generation_config = genai.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=temperature,
                    response_mime_type="application/json"
                      # Ensure output is in JSON format
                ),
            )
            num_tokens += response.usage_metadata.total_token_count
            num_completion_tokens += response.usage_metadata.prompt_token_count
            num_prompt_tokens += response.usage_metadata.candidates_token_count
    
            # checking well-formed json content
            output = response.text
            try:
                json_output = json.loads(output)
                return response, num_tokens, num_completion_tokens, num_prompt_tokens #.choices[0].message.content
            except json.JSONDecodeError as json_error:
                print(f"Malformed JSON content: {json_error}. Retrying ({ct}/3)...")
                continue  # Retry the request
        except Exception as e:
            print("Error")
            print(e)
            continue
            #logging.error(traceback.format_exc())
    return response, num_tokens, num_completion_tokens, num_prompt_tokens


"""
Helper function to iteratively evaluate prompts through the LLM (e.g., model_name) at the given temperature.

Returns the token usage and dictionary (video id to output mapping)
"""
def evaluate_prompts_gemini(client, provided_prompt, model_name, temperature):
    vid_to_output = defaultdict()
    total_completion_token, total_prompt_token = 0, 0


    # iterating through each prompt into the LLM
    for i, (vid, prompt) in enumerate(provided_prompt.items()):
        if i % 30 ==0:
            print(i)
        
        response, num_tokens, num_completion_tokens, num_prompt_tokens = get_response_gemini(client, model_name, prompt, temperature)

        # saving the outputs
        total_completion_token += num_completion_tokens
        total_prompt_token += num_prompt_tokens
        vid_to_output[vid] = response

    print("Total Number of Input Token: " + str(total_prompt_token))
    print("Total Number of Output Token: " + str(total_completion_token))
    return vid_to_output, total_completion_token, total_prompt_token


"""
Given the completion output, extracts the label from the generation.

By default, add the chat completion object returned by GPT-4, but if already extracted,
add the chat content in the completion_output parameter
"""
def extract_label(completion_output, extracted=False):
    # extracting the content & converting the string to json
    output = ""
    if extracted:
        output = completion_output.choices[0].message.content  
    else:
        output = completion_output
    try:
        json_output = json.loads(output)    
    except:
        print("parse error")
        print(output)
    
    # extracting the label
    try:
        if isinstance(json_output, list):
            json_output = json_output[0]
        
        label = str(json_output['LABEL'])
    except Exception as e:
        print("label extraction error")
        print(json_output)
        print(e)
    
    # standardizing the label
    if '-1' in label:
        return -1
    elif '0' in label:
        return 0
    elif '1' in label:
        return 1
    else:
        print("Error parsing the label")
        print(label)
        return None
    
"""
Extracts the content from the evaluation_dictionary and saves into a file called
file_name
"""
def extract_and_save_output(file_name, evaluation_dictionary):
    # extracting the content from the chat completion object
    cleaned_dictionary_to_save = dict()
    for vid, chat_completion in evaluation_dictionary.items():
        try:
            cleaned_dictionary_to_save[vid] = chat_completion.choices[0].message.content  
        except Exception as e:
            print(e)
            print(chat_completion)
            print(vid)
            cleaned_dictionary_to_save[vid] = chat_completion

    save_output(file_name, cleaned_dictionary_to_save)

"""
Extracts the content from the evaluation_dictionary and saves into a file called
file_name
"""
def extract_and_save_output_gemini(file_name, evaluation_dictionary):
    # extracting the content from the chat completion object
    cleaned_dictionary_to_save = dict()
    for vid, chat_completion in evaluation_dictionary.items():
        cleaned_dictionary_to_save[vid] = chat_completion.text

    save_output(file_name, cleaned_dictionary_to_save)
        
"""
Extracts the content from the evaluation_dictionary and saves into a file called
file_name
"""
def save_output(file_name, evaluation_dictionary):
    # saving the file
    with open(file_name, 'w') as json_file:
        json.dump(evaluation_dictionary, json_file, indent=4)