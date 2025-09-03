import os
import gc
import random
import pandas as pd
import pickle
import torch
import numpy as np
import json
import math
import argparse
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torcheval.metrics.functional import multiclass_f1_score
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score 
import wandb
random.seed(420)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--MAX_LEN", type=str, default=1024)
    parser.add_argument("--MYTH_ANNOTATION", type=str)
    parser.add_argument("--UPSAMPLE", type=bool, default=False)
    parser.add_argument("--CLASS_WEIGHT", type=bool, default=False)
    return parser.parse_args()
args = get_args()

str_modelname=args.model_name.split("/")[-1] + "-oud-" + args.MYTH_ANNOTATION 
if args.UPSAMPLE:
    str_modelname += '-UPSAMPLE'
elif args.CLASS_WEIGHT:
    str_modelname += '-CLASS_WEIGHT'

PATH_MODEL_SAVE=os.getcwd() + "/training/logs/"+f"{str_modelname}/"
if not os.path.isdir(PATH_MODEL_SAVE):
    os.mkdir(PATH_MODEL_SAVE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=PATH_MODEL_SAVE+"training.log",  # Specify the log file name
    filemode='a'  # Overwrite the log file on each run; use 'a' for appending
)

wandb.Api().create_project(str_modelname, entity='hjung-university-of-washington')

# Batch size
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
NUM_EPOCHS = 20
REPORT_EVERY = 150
min_save_epoch = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(device)

def initialize_model(modelname, bool_tokenizer=False, bool_model=True):
    # initialize tokenizer
    if bool_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, truncation=True, max_len=args.MAX_LEN, padding='max_length', cache_dir="/gscratch/scrubbed/hjung10/")
    else:
        tokenizer = None

    # initialize model
    if bool_model:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3, cache_dir="/gscratch/scrubbed/hjung10/").to(device)
    else:
        model = None
    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_training_data(TRAINING_DIR, drop_columns):
    df_data = pd.read_csv(TRAIN_DATA_DIR)
    df_data = df_data.drop_duplicates(subset='video_id')

    df_data = df_data.drop(axis=1, columns=drop_columns).reset_index()
    logging.info("Shape of dataframe: " + str(df_data.shape))
    return df_data

# Function to create weighted sampler for imbalanced classes
def create_weighted_sampler(processed_data):
    # Extract labels from processed data
    labels = [instance[1] for instance in processed_data]
    
    # Count samples per class
    class_counts = np.bincount(labels)
    logging.info(f"Class distribution: {class_counts}")
    
    # Calculate weights (inverse of frequency)
    class_weights = 1. / class_counts
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    logging.info(f"Class weights: {class_weights}")
    
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

# Function to calculate class weights for weighted loss
def calculate_class_weights(labels):
    # Count samples per class
    class_counts = np.bincount(labels)
    logging.info(f"Class distribution: {class_counts}")
    
    # Calculate weights inversely proportional to class frequency
    n_samples = len(labels)
    n_classes = len(class_counts)
    weights = n_samples / (n_classes * class_counts)
    
    # Normalize weights 
    weights = weights / weights.sum() * n_classes
    
    logging.info(f"Class weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float)

def process_input_text(row):
    title = row['video_title'] if row['video_title'] == row['video_title'] else ""
    if title == "":
        return None
    description = row['video_description'] if row['video_description'] == row['video_description'] else ""
    transcript = row['transcript'] if row['transcript'] == row['transcript'] else ""
    tags = row['tags'] if row['tags'] == row['tags'] else ""
    input = 'VIDEO TITLE: ' + title + '\nVIDEO DESCRIPTION: ' + description + '\nVIDEO TRANSCRIPT: ' + transcript + '\nVIDEO TAGS: ' + tags
    return input

### incrementing labels by 1 since prediction index starts from 0
## 0 -> opposing
### 1 -> neutral
### 2 -> supporting
def process_data_for_training(data, anno_column):
    processed = []
    for i, row  in data.iterrows():
        if i % 200 == 0:
            logging.info(i)

        # processing inputs and labels
        original_ = process_input_text(row)
        annotation = row[anno_column]
        instance = [original_, int(annotation) + 1, row['video_id']]
        
        tokenized = tokenizer(instance[0], max_length=args.MAX_LEN, padding = 'max_length', truncation=True, return_tensors='pt') #.to(device)
        if 'token_type_ids' in tokenized:
            tokenized = {'input_ids':tokenized['input_ids'][0], 'token_type_ids':tokenized['token_type_ids'][0], 'attention_mask':tokenized['attention_mask'][0]}
        else:
            tokenized = {'input_ids':tokenized['input_ids'][0], 'attention_mask':tokenized['attention_mask'][0]}
    
        instance[0]=tokenized
        processed.append(instance)
    return processed

class CustomData(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def save_list_to_json(data, filename):
    list_example = []
    for example in data_dev:
        list_example.append(example[2])
    
    """Saves a list into a JSON file."""
    with open(filename, 'w') as file:
        json.dump(list_example, file, indent=4)

def read_list_from_json(filename):
    """Reads a list from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)

def print_distribution(data_list):
    labels_to_frequency = defaultdict(int)
    for item in data_list:
        labels_to_frequency[item[1]] += 1
    logging.info(labels_to_frequency)
    
    total_freq = sum(labels_to_frequency.values())
    
    logging.info("Percentage Distribution:")
    label_to_dist = dict()
    for label, freq in labels_to_frequency.items():
        percentage = (freq / total_freq) * 100
        logging.info(f"{label}: {percentage:.2f}%")
        label_to_dist[label] = (freq / total_freq)

    return label_to_dist

# evaluation against the provided validation dataloader
def evaluate(dataloder):
    model.eval()
    val_loss, num_example, num_correct = 0, 0, 0
    all_pred = torch.Tensor().to(device)
    all_labl = torch.Tensor().to(device)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloder):
            #if i % 10 == 0:
            #    print(i)
            input, lbl, id = batch
            input = {k:v.to(device) for k,v in input.items()}
            lbl = lbl.to(device)
            outputs = model(**input, labels=lbl)
            val_loss += outputs.loss
            num_example += len(lbl)
            _, pred_idx = outputs.logits.max(dim=1)
            num_correct += sum(pred_idx == lbl).item()

            all_pred = torch.cat((all_pred, pred_idx))
            all_labl = torch.cat((all_labl, lbl))

    weighted_f1 = f1_score(all_pred.cpu(), all_labl.cpu(), average='weighted')
    macro_f1 = f1_score(all_pred.cpu(), all_labl.cpu(), average='macro')
    
    return val_loss/len(dataloder), num_correct/num_example, weighted_f1, macro_f1

# reports all the training, validation results to track
def report_train_val_results(train_loss, num_batch, num_correct_pred, num_example, dataloader_dev, bool_verbose):
    train_loss = train_loss / num_batch
    train_acc = num_correct_pred / num_example

    # validation results
    val_loss, val_acc, weighted_f1_dev, macro_f1_dev = evaluate(dataloader_dev)
   
    return train_loss, train_acc, val_loss, val_acc, weighted_f1_dev, macro_f1_dev

def train(epoch=0, best_val_f1=0, bool_verbose=True, bool_save=True, min_save_epoch=3, loss_function=None):
    model.train()
    train_loss = 0
    num_total_batch = len(dataloader_train)
    num_batch, num_example, num_correct_pred = 0, 0, 0

    for idx_batch, batch in enumerate(dataloader_train):
        if idx_batch % 50 == 0:
            print(idx_batch)

        input, lbl, video_id = batch    # contains batch_size number of tokenized inputs/labels
        input = {k:v.to(device) for k,v in input.items()}   # parsing the tokenized inputs to cuda device
        lbl = lbl.to(device)
        outputs = model(**input)  

        # tracking loss, number of examples/batch seen, and training prediction accuracy
        logits = outputs.logits
        loss = loss_function(logits, lbl)
        train_loss += loss
        num_example += len(lbl)
        num_batch += 1
        _, pred_idx = logits.max(dim=1)
        num_correct_pred += sum(pred_idx == lbl).item()

        # reporting training results and the validation results
        if (idx_batch+1) % REPORT_EVERY == 0:
            train_loss, train_acc, val_loss, val_acc, weighted_f1_dev, macro_f1_dev = report_train_val_results(train_loss, num_batch, num_correct_pred, num_example, dataloader_dev, bool_verbose)
            
            if macro_f1_dev > best_val_f1:
                best_val_f1 = macro_f1_dev
                
                if bool_verbose:
                    logging.info(f"new best val F1 found ({macro_f1_dev:.3f})")
            
            train_loss, num_batch, num_correct_pred, num_example = 0, 0, 0, 0
            model.train()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # tracking the best validation accuracy
    train_loss, train_acc, val_loss, val_acc, weighted_f1_dev, macro_f1_dev = report_train_val_results(train_loss, num_batch, num_correct_pred, num_example, dataloader_dev, bool_verbose)
    if macro_f1_dev > best_val_f1:
        best_val_f1 = macro_f1_dev
        
        if bool_verbose:
            logging.info(f"new best val F1 found ({macro_f1_dev:.3f})")
        if bool_save and epoch>=min_save_epoch:
            logging.info(f"saving the model to {PATH_MODEL_SAVE}")
            model.save_pretrained(PATH_MODEL_SAVE, from_pt=True) 
                        
    return train_loss, train_acc, val_loss, val_acc, weighted_f1_dev, macro_f1_dev, best_val_f1

# tokenizing our data
model, tokenizer = initialize_model(args.model_name, bool_tokenizer=True, bool_model=True)
TRAIN_DATA_DIR = os.getcwd() +'/training/training_data/final-search-results-annotations.csv'
columns_to_drop = ['Unnamed: 0.1', 'Unnamed: 0']
df_data = load_training_data(TRAIN_DATA_DIR, columns_to_drop)
data = process_data_for_training(df_data, args.MYTH_ANNOTATION)

# PATH TO THE TEST SET
PATH_TEST_SET = os.getcwd() + os.sep + 'test_data' + os.sep
TEST_FILE_LIST = os.listdir(PATH_TEST_SET)
TEST_FILE_LIST.remove('.ipynb_checkpoints')

# reading in the test set and getting the unique video IDs in the test set
annotated_df = pd.read_csv(PATH_TEST_SET + args.MYTH_ANNOTATION + '_evaluation_set.csv')
vid_to_label = dict()
for i, row in annotated_df.iterrows():
    vid_to_label[row['video_id']] = row['label']
annotated_vid = set(annotated_df['video_id'].tolist())

# Videos to exclude as they are included in the few-shot example
MYTH_TO_FEW_SHOT_EXCLUDE = {
    'M1' : ['SjCZwqEE22Y', '7PT0gv6a97o', 'X3UKcHR-2uU', 'fTcGyWDDg5s', 'bMitni3tC-c'],
    'M2' : ['9TYr6sqDEfY', 'DyjRxf-aJN4', 'AnUN2Zs4Mnk', 'm_uV8UkTDKw', '-3G162dqVVI'],
    'M3' : ['Jc-buPCKisM', '0hR2Hwkhey8', 'UfQWOGOFNFA', 'JczoO7ogOS8', 'SjCZwqEE22Y'],
    'M4' : ['DyjRxf-aJN4', '7PT0gv6a97o', 'Qg7G0VTi3iY', 'OFGFeA6Ap7E', 'v4GnSSvcYys'],
    'M5' : ['zN9NDZ6lgaM', '7PT0gv6a97o', 'sZ5-i72Yl2Q', 'FmGalSsq63k', 'TnYHKxUHgCs'],
    'M6' : ['tzHKfZyevXo', 'eWdCJm9q1bw', '7gtWuoWGQWM', 'WNXieqey_iQ', 'SjCZwqEE22Y'],
    'M7' : ['QtRQ9UD7rpY', 'W-7_alg4I28', '0RkpSTlvvj0', '0y55ymuJ2K4', 'TP0ToVYXQ-k'],
    'M10' : ['6F6d10ggVDw', 'nmMCQ1y8l14', 'GI3blNNe56w', 'E9jKyHjPbUg', 'QtRQ9UD7rpY']
}

annotated_vid = annotated_vid - set(MYTH_TO_FEW_SHOT_EXCLUDE[args.MYTH_ANNOTATION])


# shuffling data
random.shuffle(data)

# ensuring that we are excluding the test set (which should also have excluded the few-shot example from our LLM eval)
data_test = []
data_train = []
for item in data:
    if item[2] in annotated_vid: 
        item[1] = vid_to_label[item[2]] + 1  # ensuring that the test set reflects the expert annotated labels rather than GPT-4o synthetic labels
        data_test.append(item)
    else:
        data_train.append(item)
logging.info("Train + Val Split: " + str(len(data_train)))

# further splitting the data into 80:20 train-validation split
file_name = os.getcwd() + os.sep + 'validation-split' + os.sep + args.MYTH_ANNOTATION + '-validation-set.jsonl'
validation_set_list = read_list_from_json(file_name)

data_dev, data_train_ = [], []
for item in data_train:
    if item[2] in validation_set_list:
        data_dev.append(item)
    else:
        data_train_.append(item)
logging.info("validation split length: " + str(len(data_dev)))
logging.info("train split length: " + str(len(data_train_)))

# Create weighted sampler for imbalanced classes
dataloader_train = ""
if args.UPSAMPLE:   # if upsample is true
    logging.info("Upsampling")
    train_sampler = create_weighted_sampler(data_train_)
    dataloader_train = DataLoader(CustomData(data_train_), batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler)
else:
    dataloader_train = DataLoader(CustomData(data_train_), batch_size=TRAIN_BATCH_SIZE, shuffle=True)

logging.info(f"data size\ttrain:{len(data_train_)}\tdev:{len(data_dev)}\ttest:{len(data_test)}")
dataloader_dev = DataLoader(CustomData(data_dev), batch_size=VALID_BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(CustomData(data_test), batch_size=VALID_BATCH_SIZE, shuffle=True)

print_distribution(data_train_)
print_distribution(data_test)


LEARNING_RATE_LIST = [5.00E-06, 1.00E-06, 1.00E-05]
WEIGHT_DECAY_LIST = [5.00E-04, 1.00E-04, 5.00E-05]

hyperparam_to_results = dict()

for LEARNING_RATE in LEARNING_RATE_LIST:
    for WEIGHT_DECAY in WEIGHT_DECAY_LIST:
        model, tokenizer = initialize_model(args.model_name, bool_tokenizer=False, bool_model=True)

        hyperparam_string = "lr:"+str(LEARNING_RATE)+',wd:'+str(WEIGHT_DECAY)
        logging.info(hyperparam_string)

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb entity where your project will be logged (generally your team name)
            entity="hjung-university-of-washington",
        
            # set the wandb project where this run will be logged
            project=str_modelname,
        
            name=hyperparam_string,
        
            # track hyperparameters and run metadata
            config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": args.model_name.split('/')[1] + '-' + args.MYTH_ANNOTATION,
            "epochs": NUM_EPOCHS,
            }
        )

        loss_function = ""
        if args.CLASS_WEIGHT:
            # Check class distribution in training data
            train_labels = [item[1] for item in data_train_]
            
            # Calculate class weights for loss function
            class_weights = calculate_class_weights(train_labels)
            logging.info(class_weights)
            loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        #metrics
        best_val_acc_tup = (0, 0)
        best_val_f1_tup = (0, 0)
                
        best_val_f1 = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc, val_loss, val_acc, weighted_f1_dev, macro_f1_dev, best_val_f1 = train(epoch, best_val_f1, bool_save=True, loss_function=loss_function)
            gc.collect()
            torch.cuda.empty_cache()

            # keeping track of best performances
            if val_acc > best_val_acc_tup[0]:
                best_val_acc_tup = (val_acc, epoch) 
            if macro_f1_dev > best_val_f1_tup[0]:
                best_val_f1_tup = (macro_f1_dev, epoch)
        
            if epoch >= min_save_epoch:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                           "train_acc": train_acc, "val_acc": val_acc, 
                           "macro_f1_dev": macro_f1_dev})

        hyperparam_to_results[hyperparam_string] = {'best_val_acc' : best_val_acc_tup,
                                                    'best_val_f1' : best_val_f1_tup}
                    
        # finish the wandb run
        wandb.finish()
        gc.collect()
        torch.cuda.empty_cache()

logging.info(hyperparam_to_results)