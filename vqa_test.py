import json
import cv2
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import pickle
import torch.nn as nn
import torch.optim as optim

print("Welcome to Antriksh How can I help you")
logging.getLogger("transformers").setLevel(logging.ERROR)
torch.hub.set_dir(r"C:\Users\ASUS\TORCH")
img_path=[]
for i in range (0,1000):
    img_path.append(i)

# Load the datasets from JSON files
with open("all_questions.json", "r") as q_file, open("all_answers.json", "r") as a_file:
    questions_data = json.load(q_file)
    answers_data = json.load(a_file)

# Create mappings for questions, answers, and image paths
image_paths = [f"{img_path[i]}.TIF" for i in range(0, 1000)]
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

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.resize(image, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def preprocess_text(question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(question, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    return tokens

def extract_image_features(image_tensor):
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    model.eval()
    with torch.no_grad():
        features = model(image_tensor)
        features = features.flatten(start_dim=1)
    return features

def extract_text_features(tokens):
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    with torch.no_grad():
        output = model(**tokens)
    cls_features = output.last_hidden_state[:, 0, :]
    return cls_features

class VQAModel(nn.Module):
    def __init__(self, num_classes=10):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(1768, 512)  # Adjust input features if necessary
        self.fc2 = nn.Linear(512, num_classes)  # Output the correct number of classes

    def forward(self, image_features, text_features):
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = torch.relu(self.fc1(combined_features)) # to remove all negative outputs 
        output = self.fc2(x)  # No activation function here, logits are directly output
        return output

def predict(model, image_path, question):
    model.eval()
    
    image_tensor = preprocess_image(image_path)
    tokens = preprocess_text(question)
    
    image_features = extract_image_features(image_tensor)
    text_features = extract_text_features(tokens)
    
    with torch.no_grad():
        logits = model(image_features, text_features)
    
    return logits

def predicted_output(logits):
    probs = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1)
    return predicted_class

def decode_prediction(predicted_class, text_labels):
    inverse_mapping = {v: k for k, v in text_labels.items()}
    return inverse_mapping.get(predicted_class.item(), "Sorry I am not too trained yet")


def generate_sentence(question, prediction):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    if prediction.lower()=="yes":
     input_text = f"  Question: {question} the Answer is: {prediction}."
    else:
         input_text =f"  Question: {question} the Answer is: {prediction}."
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids,max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



while True:
    image_path = input("ENTER THE NAME OF THE IMAGE WITH EXTENSION: ")
    question = input("ENTER THE TEXT HERE: ")
    
    # Load the trained VQA model
    model = VQAModel(num_classes=2)
    model.load_state_dict(torch.load('vqa_model_classification.pth'))
    model.eval()

    # Predict the answer
    logits = predict(model, image_path, question)
    predicted_index = predicted_output(logits)
    predicted_answer = decode_prediction(predicted_index, label_mapping)

    FINAL_OUTPUT=generate_sentence(question,predicted_answer)
    print(FINAL_OUTPUT)

    
    choice = input("DO YOU WANT TO PROCEED [Y/N] ")
    if choice.lower() != 'y':
        break
