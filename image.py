import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, BlipProcessor, BlipForConditionalGeneration, pipeline
import easyocr
import numpy as np
import cv2
import pandas as pd

# Load models and processors
segment_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
segment_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
ocr_reader = easyocr.Reader(['en'])
summarizer = pipeline("summarization")

# Streamlit UI
st.title("AI Pipeline for Image Segmentation and Object Analysis")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Step 1: Image Segmentation
    st.write("**Step 1: Image Segmentation**")
    inputs = segment_processor(images=image, return_tensors="pt")
    outputs = segment_model(**inputs)
    panoptic_seg = segment_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    segmentation_map = panoptic_seg['segmentation'].cpu().numpy()
    segments_info = panoptic_seg['segments_info']

    # Display segmentation results
    unique_labels = np.unique(segmentation_map)
    colored_segmentation = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    for label in unique_labels:
        mask = segmentation_map == label
        color = np.random.randint(0, 255, size=3)
        colored_segmentation[mask] = color
    st.image(colored_segmentation, caption='Segmented Image', use_column_width=True)

    # Step 2: Object Extraction and Identification
    st.write("**Step 2: Object Extraction and Identification**")
    object_descriptions = []
    object_images = []
    
    for segment in segments_info:
        mask = (segmentation_map == segment["id"]).astype("uint8")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            extracted_object = np.array(image)[y:y+h, x:x+w]
            extracted_object_img = Image.fromarray(extracted_object)
            object_images.append(extracted_object_img)
            
            # Generate caption for the object
            inputs = caption_processor(extracted_object_img, return_tensors="pt")
            out = caption_model.generate(**inputs)
            caption = caption_processor.decode(out[0], skip_special_tokens=True)
            
            object_descriptions.append({
                "Object": f"Object {segment['id']}",
                "Description": caption
            })

    # Display object descriptions
    description_df = pd.DataFrame(object_descriptions)
    st.table(description_df)

    # Step 3: Text Extraction
    st.write("**Step 3: Text Extraction**")
    object_text_data = []
    for idx, extracted_object_img in enumerate(object_images):
        extracted_text = ocr_reader.readtext(np.array(extracted_object_img), detail=0)
        extracted_text = ' '.join(extracted_text).strip()
        object_text_data.append({
            "Object Image": f"Object {idx+1}",
            "Extracted Text": extracted_text
        })
    
    # Display extracted text
    text_data_df = pd.DataFrame(object_text_data)
    st.table(text_data_df)

    # Step 4: Summarize Object Attributes
    st.write("**Step 4: Summarize Object Attributes**")
    object_summaries = []
    for index, row in description_df.iterrows():
        description = row['Description']
        summary = summarizer(description, max_length=50, min_length=25, do_sample=False)
        object_summaries.append({
            "Object Image": row['Object'],
            "Summary": summary[0]['summary_text']
        })
    
    summaries_df = pd.DataFrame(object_summaries)
    st.table(summaries_df)

    # Step 5: Output Generation
    st.write("**Final Output**")
    # Here you should include the code to generate the annotated image
    # Since it's not fully provided, I will just display the summaries table
    st.table(summaries_df)
