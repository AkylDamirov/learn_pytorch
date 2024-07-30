import gradio as gr
import os
import torch

from demos.foodvision_mini2.model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation ###
# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(len(class_names))

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        '09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
        map_location=torch.device('cpu')
    )
)

### 3. Predict function ###
# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)

        # Create a prediction label and prediction probability dictionary for each prediction class
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

        # Calculate the prediction time
        pred_time = round(timer() - start_time, 5)

        # Return the prediction dictionary and prediction time
        return pred_labels_and_probs, pred_time

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create examples list from 'examples/' directory
example_list = [['examples/' + example] for example in os.listdir('examples')]

# Create gradio demo
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Label(num_top_classes=3, label='Predictions'),
        gr.Number(label='Prediction time (s)')
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article
)

# Launch the demo
demo.launch()


