import json
import cv2
import torch
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import pickle
import torch.nn as nn
import torch.optim as optim

# Set logging level and directory for torch
logging.getLogger("transformers").setLevel(logging.ERROR)
torch.hub.set_dir(r"C:\Users\ASUS\TORCH")
print("LOADING THE DATASET")
img_path=[]
for i in range (0,772):
    img_path.append(i)
# Load the datasets from JSON files
with open("all_questions.json", "r") as q_file, open("all_answers.json", "r") as a_file:
    questions_data = json.load(q_file)
    answers_data = json.load(a_file)

# Create mappings for questions, answers, and image paths
image_paths = [f"{img_path[i]}.TIF" for i in range(0, 772)]
questions=[]
question_ids=[]
# Create mappings for questions, answers, and image paths
for question in  questions_data['questions']:
    if  question['type']=='presence' or question['type']=='comp'  :
        questions.append(question['question'])
        question_ids.append(question['id'])
# Map question IDs to their corresponding answers
answer_mapping = {answer['question_id']: answer['answer'] for answer in answers_data['answers']}
text_labels = [answer_mapping[qid] for qid in question_ids if qid in answer_mapping]

# Create a mapping from label index to answer
label_mapping = {label: idx for idx, label in enumerate(set(text_labels))}
processed_labels = [label_mapping[label] for label in text_labels]

print(len(questions))
print(processed_labels)