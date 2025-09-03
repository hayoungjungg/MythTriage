import tiktoken
from utils import GPTRequests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

"""
Constructs the provided prompt given the video metadata in the df for zero-shot.

Provide None value for few_shot parameter if zero_shot. Provide few_shot examples otherwise
"""
def create_myth_specific_prompts(df, prompt, persona, myth, few_shot):
    id_to_prompt = dict()
    for i, row in df.iterrows():
        new_prompt = str(prompt)
        id = row['video_id']
        title = row['video_title'] if row['video_title'] == row['video_title'] else ""
        description = row['video_description'] if row['video_description'] == row['video_description'] else ""
        transcript = row['transcript'] if row['transcript'] == row['transcript'] else ""
        tags = row['tags'] if row['tags'] == row['tags'] else ""

        if title == "":
            continue

        new_prompt = new_prompt.replace('[TITLE]', title)
        new_prompt = new_prompt.replace('[DESCRIPTION]', description)
        new_prompt = new_prompt.replace('[TRANSCRIPT]', transcript)
        new_prompt = new_prompt.replace('[TAGS]', tags)
        new_prompt = new_prompt.replace('[MYTH-DEFINITION]', myth)
        if few_shot:
            new_prompt = new_prompt.replace('[FEW-SHOT EXAMPLE]', few_shot)
        id_to_prompt[id] = [{"role": "system", "content": persona}, {"role": "user", "content": f"{new_prompt}"}]
        
    total_token, avg_tokens = average_input_tokens(id_to_prompt)
    print("The total input tokens: " + str(total_token))
    print("The average input tokens: " + str(avg_tokens))
        
    return id_to_prompt

"""
Constructs the provided prompt given the video metadata in the df for zero-shot. FOR CLAUDE

Provide None value for few_shot parameter if zero_shot. Provide few_shot examples otherwise
"""
def create_myth_specific_prompt_claude_batches(df, prompt, persona, myth, model_name, few_shot):
    requests_list = []
    for i, row in df.iterrows():
        new_prompt = str(prompt)
        id = row['video_id']
        title = row['video_title'] if row['video_title'] == row['video_title'] else ""
        description = row['video_description'] if row['video_description'] == row['video_description'] else ""
        transcript = row['transcript'] if row['transcript'] == row['transcript'] else ""
        tags = row['tags'] if row['tags'] == row['tags'] else ""

        if title == "":
            continue

        new_prompt = new_prompt.replace('[TITLE]', title)
        new_prompt = new_prompt.replace('[DESCRIPTION]', description)
        new_prompt = new_prompt.replace('[TRANSCRIPT]', transcript)
        new_prompt = new_prompt.replace('[TAGS]', tags)
        new_prompt = new_prompt.replace('[MYTH-DEFINITION]', myth)
        if few_shot:
            new_prompt = new_prompt.replace('[FEW-SHOT EXAMPLE]', few_shot)
            
        req_object = Request(
            custom_id=id,
            params=MessageCreateParamsNonStreaming(
            model=model_name,
            max_tokens=1024,
            system=persona,
            messages=[{
                "role": "user",
                "content": f"{new_prompt}",
            }]))
        requests_list.append(req_object)
        
    return requests_list

"""
Constructs the provided prompt given the video metadata in the df for zero-shot.

Provide None value for few_shot parameter if zero_shot. Provide few_shot examples otherwise
"""
def create_myth_specific_prompts_gemini(df, prompt, persona, myth, few_shot):
    id_to_prompt = dict()
    for i, row in df.iterrows():
        new_prompt = str(prompt)
        id = row['video_id']
        title = row['video_title'] if row['video_title'] == row['video_title'] else ""
        description = row['video_description'] if row['video_description'] == row['video_description'] else ""
        transcript = row['transcript'] if row['transcript'] == row['transcript'] else ""
        tags = row['tags'] if row['tags'] == row['tags'] else ""

        if title == "":
            continue

        new_prompt = new_prompt.replace('[TITLE]', title)
        new_prompt = new_prompt.replace('[DESCRIPTION]', description)
        new_prompt = new_prompt.replace('[TRANSCRIPT]', transcript)
        new_prompt = new_prompt.replace('[TAGS]', tags)
        new_prompt = new_prompt.replace('[MYTH-DEFINITION]', myth)
        if few_shot:
            new_prompt = new_prompt.replace('[FEW-SHOT EXAMPLE]', few_shot)
        id_to_prompt[id] = persona + "\n\n" + new_prompt
                
    return id_to_prompt

def truncate_prompt(prompt: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """
    Truncate the input prompt to a specified maximum token length for LLMs.

    Args:
        prompt (str): The input prompt to truncate.
        max_tokens (int): The maximum allowable token count for the prompt.
        model (str): The model name to determine the tokenizer (default: "gpt-3.5-turbo").

    Returns:
        str: The truncated prompt.
    """
    # Load the tokenizer for the specified model
    tokenizer = tiktoken.encoding_for_model(model)

    # Tokenize the prompt
    tokens = tokenizer.encode(prompt)

    # Truncate the tokens if necessary
    truncated_tokens = tokens[:max_tokens]

    # Decode the tokens back into a string
    truncated_prompt = tokenizer.decode(truncated_tokens)

    return truncated_prompt


"""
Constructs the provided prompt given the video metadata in the df for zero-shot.

Provide None value for few_shot parameter if zero_shot. Provide few_shot examples otherwise
"""
def create_myth_specific_prompts_qwen(df, prompt, persona, myth, few_shot, max_token):
    id_to_prompt = dict()
    for i, row in df.iterrows():
        new_prompt = str(prompt)
        id = row['video_id']
        title = row['video_title'] if row['video_title'] == row['video_title'] else ""
        description = row['video_description'] if row['video_description'] == row['video_description'] else ""
        transcript = row['transcript'] if row['transcript'] == row['transcript'] else ""
        tags = row['tags'] if row['tags'] == row['tags'] else ""

        if title == "":
            continue

        new_prompt = new_prompt.replace('[TITLE]', title)
        new_prompt = new_prompt.replace('[DESCRIPTION]', description)
        new_prompt = new_prompt.replace('[TRANSCRIPT]', transcript)
        new_prompt = new_prompt.replace('[TAGS]', tags)
        new_prompt = new_prompt.replace('[MYTH-DEFINITION]', myth)
        if few_shot:
            new_prompt = new_prompt.replace('[FEW-SHOT EXAMPLE]', few_shot)
            
        new_prompt = truncate_prompt(new_prompt, max_token - 500)
        id_to_prompt[id] = [{"role": "system", "content": persona}, {"role": "user", "content": f"{new_prompt}"}]
        
    total_token, avg_tokens = average_input_tokens(id_to_prompt)
    print("The total input tokens: " + str(total_token))
    print("The average input tokens: " + str(avg_tokens))
        
    return id_to_prompt


"""
Calculates the average input tokens to estimate cost and decide which model to use
"""
def average_input_tokens(prompts, model_name="gpt-3.5-turbo"):
    # Load the encoding for the model
    encoding = tiktoken.encoding_for_model(model_name)
    
    # Calculate the number of tokens for each prompt
    token_counts = [len(encoding.encode(prompt[1]['content'])) for prompt in prompts.values()]
    
    # Calculate the average token count
    total_token = sum(token_counts)
    avg_tokens = total_token / len(prompts) if prompts else 0
    
    return total_token, avg_tokens

"""
Function to compute the performance results compared to the ground-truth labels

Input:
- vid_to_output: Dictionary containing the prediction results
- vid_to_label_eval: Dictionary containing the ground truth labels
- chat_completion_bool: True if Chat Completion object, False otherwise
- list_video_id_exclude: list of video ids to exclude from the result computation
"""
def compute_results(vid_to_output, vid_to_label_eval, chat_completion_bool, list_video_id_exclude):
    # extracting the labels and organizing into list
    predictions = []
    ground_truths = []
    for vid, output in vid_to_output.items():
        if vid not in list_video_id_exclude:
            if type(output) == int:
                predictions.append(output)
            else:
                predictions.append(GPTRequests.extract_label(output, chat_completion_bool))
            ground_truths.append(vid_to_label_eval[vid])

    # computing accuracy, macro and weighted F1-scores, precision and recall
    accuracy = accuracy_score(ground_truths, predictions)
    f1_macro = f1_score(ground_truths, predictions, average='macro')
    f1_weighted = f1_score(ground_truths, predictions, average='weighted')

    # Compute precision and recall (macro average)
    precision = precision_score(ground_truths, predictions, average='macro')
    recall = recall_score(ground_truths, predictions, average='macro')

    # Generate classification report
    class_report = classification_report(ground_truths, predictions)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(ground_truths, predictions)

    # Print metrics
    print("Accuracy:", round(accuracy, 3))
    print("F1-Score (Macro):", round(f1_macro, 3))
    print("F1-Score (Weighted):", round(f1_weighted, 3))
    print("Precision (Macro):", round(precision, 3))
    print("Recall (Macro):", round(recall, 3))