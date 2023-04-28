### 1. Imports and class names setup ###
import gradio as gr
import os 
import torch
import torchvision

from model import create_densenet161_model
from model import create_vit_model
from model import create_effnetb2_model
from model import MyEnsemble
from torchvision import transforms

from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']

### 2. Model and transforms perparation ###
densenet161, densenet161_transforms = create_densenet161_model(num_classes=3)
vit, vit_transforms = create_vit_model()
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3)
model = MyEnsemble(densenet161, vit, effnetb2, 3)
# Create image size 
IMG_SIZE = 224 # comes from Table 3 of the ViT paper

# Create transforms pipeline
manual_transforms = transforms.Compose([
                                        transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                        transforms.ToTensor()
])

# Load save weights
os.system('wget -P /home/xlab-app-center/ https://huggingface.co/spaces/ananya0409/Lung_Cancer_Detection/resolve/main/lung_ensemble_densenet_vit_effnetb2.pth')

model.load_state_dict(
    torch.load(
        f="lung_ensemble_densenet_vit_effnetb2.pth",
        map_location=torch.device("cpu") # load the model to the CPU
    )
)

### 3. Predict function ### 

def predict(img) -> Tuple[Dict, float]:
  # Start a timer
  start_time = timer()
  # Transform the input image for use with model
  img = manual_transforms(img).unsqueeze(0) # unsqueeze = add batch dimension on 0th index

  # Put model into eval mode, make prediction
  model.eval()
  with torch.inference_mode():
    # Pass transformed image through the model and turn the prediction logits into probaiblities
    pred_probs = torch.softmax(model(img), dim=1)

  # Create a prediction label and prediction probability dictionary
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate pred time
  end_time = timer()
  pred_time = round(end_time - start_time, 4)

  # Return pred dict and pred time
  return pred_labels_and_probs, pred_time

### 4. Gradio app ### 

# Create title, description and article
title = "Lung Cancer Detection"
description = "An [ensemble feature extractor] AI model to classify images as benign, normal and malignant lung CT images."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/#74-building-a-gradio-interface)."

# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]
# Create the Gradio demo
demo = gr.Interface(fn=predict, # maps inputs to outputs
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch() 
