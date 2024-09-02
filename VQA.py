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
image_paths = [f"{img_path[i]}.TIF" for i in range (0,772)]
questions=[]
question_ids=[]
# Create mappings for questions, answers, and image paths
for question in  questions_data['questions']:
    if  question['type']=='count':
        questions.append(question['question'])
        question_ids.append(question['id'])

# Map question IDs to their corresponding answers
answer_mapping = {answer['question_id']: answer['answer'] for answer in answers_data['answers']}
text_labels = [answer_mapping[qid] for qid in question_ids]

# Create a mapping from label index to answer
label_mapping = {label: idx for idx, label in enumerate(set(text_labels))}
processed_labels = [label_mapping[label] for label in text_labels]
# Define preprocessing functions
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

# Define the VQA model
class VQAModel(nn.Module):
     def __init__(self, num_classes=10):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(1768, 512)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.fc2 = nn.Linear(512, num_classes)

     def forward(self, image_features, text_features):
        combined_features = torch.cat((image_features, text_features), dim=1)
        x = torch.relu(self.fc1(combined_features))
        x = self.dropout(x)  # Apply dropout
        output = self.fc2(x)
        return output


# Process images and text
image_features = {}
text_features = []
for image_path, question in zip(image_paths, questions):
    try:
        image_tensor = preprocess_image(image_path)
        img_features = extract_image_features(image_tensor)
        image_features[image_path] = img_features.squeeze(0).numpy()  # Store the feature vector
        
        tokens = preprocess_text(question)
        txt_features = extract_text_features(tokens)
        text_features.append(txt_features.squeeze(0).numpy())  # Store the feature vector
    except FileNotFoundError as e:
        print(e)

# Save image features
with open('image_features.pkl', 'wb') as f:
    pickle.dump(image_features, f)

# Define the VQA dataset
class VQADataset(Dataset):
    def __init__(self, image_features, questions, labels):
        self.image_features = image_features  # Dictionary with image paths as keys
        self.questions = questions
        self.labels = labels
        
        # Create a list of image keys corresponding to each question
        self.image_keys = [list(self.image_features.keys())[i % len(self.image_features)] for i in range(len(self.questions))]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Get the image key for the current question
        img_key = self.image_keys[idx]
        
        # Access the image feature using the image key
        image_feature = self.image_features[img_key]
        question = self.questions[idx]
        label = self.labels[idx]
        
        return image_feature, question, label

# Instantiate the dataset and dataloader
dataset = VQADataset(image_features, text_features, processed_labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the model, loss function, and optimizer
model = VQAModel(num_classes=len(label_mapping))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
import time

num_epochs = 30
# Adjust learning rate and batch size
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Implement learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in dataloader:
        if batch is None:
            continue
        
        image_features_batch, text_features_batch, labels_batch = batch
        
        optimizer.zero_grad()

        outputs = model(image_features_batch, text_features_batch)

        loss = criterion(outputs, torch.tensor(labels_batch, dtype=torch.long))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels_batch).sum().item()
        total_predictions += labels_batch.size(0)
    
    scheduler.step()  # Update learning rate

    epoch_time = time.time() - start_time
    accuracy = correct_predictions / total_predictions
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s")
print(accuracy)
# Save the model
torch.save(model.state_dict(), 'vqa_model_maths.pth')

# Define prediction functions
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

def decode_prediction(predicted_class, label_mapping):
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    return inverse_mapping.get(predicted_class.item(), "Sorry I am not too trained yet")

# Example usage
image_path = '705.TIF'
question = "Is there a water body in the image?"

# Load the trained model
model = VQAModel(num_classes=len(label_mapping))
model.load_state_dict(torch.load('vqa_model_mathematical.pth'))
model.eval()

# Predict the answer
logits = predict(model, image_path, question)
predicted_index = predicted_output(logits)
predicted_answer = decode_prediction(predicted_index, label_mapping)
print("Predicted Answer:", predicted_answer)
