import configparser
import streamlit as st

import matplotlib.pyplot as plt


from src.mlp import MLP
from src.toy_colors_dataset import generate_pytorch_dataset, generate_dataloaders
from src.train import test_loop
from src.explain import explain




args = configparser.ConfigParser()
args.read('src/argsConfig.ini')

train_dataset, dev_dataset, test_dataset = generate_pytorch_dataset(args)
_, _, test_iterator = generate_dataloaders(args, train_dataset, dev_dataset, test_dataset)
model = MLP(args)
optimizer = model.get_optimizer()
criterion = model.get_loss_function()
device = model.get_device()

model = model.to(device)
criterion = criterion.to(device)

class1, class2, incorrect_examples = test_loop(args, model, test_iterator, criterion, device)


st.set_page_config(page_title='Toycolors Dataset Predict & Explain:', page_icon='random', layout='centered', initial_sidebar_state='collapsed')
st.title('Toycolors Dataset Predict & Explain:')
st.text('Select Class to Explain:')
class_in = st.selectbox('Models:', ['Class-1', 'Class-2', 'Incorrect-Predictions'], index=0)


if class_in == "Class-1":
    index = st.number_input("Index of Class 1 examples", min_value=1, max_value=len(class1))
    example = class1[index - 1]
    fig, ax = explain("LayerLRP", model, example)
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)
    st.pyplot(fig)


elif class_in == "Class-2":
    index = st.number_input("Index of Class 2 examples", min_value=1, max_value=len(class2))
    example = class2[index - 1]
    fig, ax = explain("LayerLRP", model, example)
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)
    st.pyplot(fig)

elif class_in == "Incorrect-Predictions":
    index = st.number_input("Index of Incorrect-Predictions examples", min_value=1, max_value=len(incorrect_examples))
    example = incorrect_examples[index - 1]
    fig, ax = explain("LayerLRP", model, example)
    fig.patch.set_facecolor('#E0E0E0')
    fig.patch.set_alpha(0.7)
    st.pyplot(fig)

else:
    raise Exception("No Valid Choice !!")





