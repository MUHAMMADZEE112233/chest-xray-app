import torch
import streamlit as st

from PIL import Image

from datasets import load_dataset
from transformers import BlipForConditionalGeneration, AutoProcessor, AutoConfig
from transformers import AutoModelForCausalLM


blip_config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-large")
gitbase_config = AutoConfig.from_pretrained("microsoft/git-large-textcaps")

device = "cpu"

blip_model = BlipForConditionalGeneration(blip_config)
gitbase_model = AutoModelForCausalLM.from_config(gitbase_config)

blip_model.load_state_dict(torch.load("blip-image-captioning-large.pth",
                                 map_location=torch.device("cpu")))

gitbase_model.load_state_dict(torch.load("git-large-textcaps.pth",
                                 map_location=torch.device("cpu")))

blip_processor = AutoProcessor.from_pretrained("ybelkada/blip-image-captioning-base-football-finetuned")
gitbase_processor = AutoProcessor.from_pretrained("microsoft/git-base")


st.title("Chest X-ray Results")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

option = st.selectbox(
    'Select your model?',
    ('blip-image', 'git-large-textcaps'))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if option == 'blip-image':
        inputs = blip_processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif option == 'git-large-textcaps':
        inputs = gitbase_processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        generated_ids = gitbase_model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = gitbase_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_caption)
    st.success(generated_caption)
