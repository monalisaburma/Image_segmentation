# AI Pipeline for Image Segmentation and Object Analysis

## Overview
This project is designed to develop a comprehensive AI pipeline that performs advanced image analysis. The pipeline processes input images by segmenting objects, identifying them, extracting relevant data, and generating a detailed summary. The final output integrates these insights into a visually annotated image and a structured summary table, providing a powerful tool for understanding and interpreting complex image content.

## Requirements
To run this project, you'll need to install the required Python libraries. You can install them using the following command:
```bash
pip install -r requirements.txt
```

## Files Included
`Image_segmentation.ipynb`: The main Jupyter notebook containing the code for image segmentation, object extraction, identification, and analysis.

`requirements.txt`: A file containing the list of Python libraries required to run the notebook.

Getting Started
1. Clone the repository:
```bash
git clone https://github.com/monalisaburma/Image_segmentation.git
```
2. Navigate to the project directory:
```bash
cd Image_Segmentation
```
3. Install the required libraries:
```bash
pip install -r requirements.txt
```

## Project Details
This project involves several key steps:

- Image Segmentation: Implemented using pre-trained models to accurately segment objects within an image.
- Object Extraction and Storage: Extracted the segmented objects, saved them as separate images, and stored their metadata in a structured format.
- Object Identification: Used an image captioning model to generate descriptions for each object, identifying and labeling them effectively.
- Text/Data Extraction (OCR): Applied Optical Character Recognition (OCR) to extract any textual data from the identified objects.
- Summarization and Data Mapping: Generated concise summaries of each object's attributes and mapped the data into a structured format.
- Output Generation: Produced a final annotated image and a summary table that clearly visualizes the results of the pipeline.

## Usage
To use the pipeline, upload an image into the Jupyter notebook, and the pipeline will automatically process the image to segment, identify, and extract information from each object within the image. The final output will include a visually annotated image and a structured summary table.

## Results
The AI pipeline successfully segmented and identified objects within the provided images, extracted relevant data, and generated detailed visual and tabular summaries. The results demonstrate the effectiveness of the pipeline in providing insights into complex image content.

## Challenges
- Model Performance: Fine-tuning models for accurate segmentation and identification.
- Data Mapping Complexity: Ensuring accurate mapping of extracted data to the correct objects.
- Resource Limitations: The project required significant computational resources for processing.
- OCR Accuracy: The OCR struggled with certain text types and required fine-tuning.

## Future Work
- Model Refinement: Further fine-tuning of the models for better accuracy.
- Extended Capabilities: Expanding the pipeline to handle more complex image analysis tasks and integrating additional data extraction methods.

## Author
[Monalisa Burma] 
[monalisaburma@gmail.com]

