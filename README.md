# COCO-CODE
A code used to detect protein crystals in images, then draw bounding boxes and masks over them.
This project involves using machine learning models to analyze images of protein crystals.
This code is named COCO code because it executes based on a COCO model.
Its main function is detecting the protein crystals in the images. It then draws bounding boxes around them and a masks over them.
It performs object detection and instance segmentation.

The code is displayed below.

# Install dependencies
!pip install -U pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install 'detectron2@git+https://github.com/facebookresearch/detectron2.git'

# Verify Detectron2 installation
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

print("Detectron2 installed successfully!")

# Import necessary libraries
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
import torch
import cv2
import datetime
import matplotlib.pyplot as plt

# Paths to your COCO format dataset
data_root = "/content/drive/MyDrive/ENM401.v1-protein-crystals.coco" **REPLACE WITH YOUR DATASET PATH**
train_path = os.path.join(data_root, "train")
train_ann_path = os.path.join(train_path, "_annotations.coco.json")

valid_path = os.path.join(data_root, "valid")
valid_ann_path = os.path.join(valid_path, "_annotations.coco.json")

test_path = os.path.join(data_root, "test")
test_ann_path = os.path.join(test_path, "_annotations.coco.json")

# Register train, validation, and test datasets with Detectron2
register_coco_instances("protein_crystals_train", {}, train_ann_path, train_path)
register_coco_instances("protein_crystals_valid", {}, valid_ann_path, valid_path)
register_coco_instances("protein_crystals_test", {}, test_ann_path, test_path)

# Print registered datasets to confirm
print("Registered datasets: ", DatasetCatalog.list())

# Create configuration object for training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("protein_crystals_train",)
cfg.DATASETS.TEST = ("protein_crystals_valid",)
cfg.TEST.EVAL_PERIOD = 1000  # Reduce evaluation frequency to every 1000 iterations 
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Number of classes (Size-1, Size-2, Size-3) **SET ACCORDING TO THE NUMBER OF CLASSES IN YOUR DATASET**
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for inference
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 800
cfg.SOLVER.STEPS = [600]
cfg.SOLVER.GAMMA = 0.1
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
cfg.OUTPUT_DIR = "/content/drive/MyDrive/COCO_Training_Output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Save the configuration to a file
cfg_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(cfg_path, 'w') as f:
    f.write(cfg.dump())
print(f"Configuration saved to: {cfg_path}")

# Train the model
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        from detectron2.evaluation import COCOEvaluator
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Inference on new images
model_dir = cfg.OUTPUT_DIR  # Path to the training output directory
cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")  # Path to the trained model weights

# Create predictor for inference
predictor = DefaultPredictor(cfg)

# Define the input folder containing images for inference
input_folder_path = "/content/drive/MyDrive/New"  # Replace this with your input folder path **SET THE INPUT FOLDER PATH FOR IMAGE INFERENCE**

# Create the output folder for saving inference results
output_folder_path = "/content/drive/MyDrive/COCO_Output_Images" **SET YOUR OUTPUT FOLDER FOR RESULT**
os.makedirs(output_folder_path, exist_ok=True)

# Loop through all files in the input folder and perform inference
for image_name in os.listdir(input_folder_path):
    # Construct the complete image path
    image_path = os.path.join(input_folder_path, image_name)

    # Ensure that we are working only with image files
    if not (image_path.endswith(".jpg") or image_path.endswith(".png") or image_path.endswith(".jpeg") or image_path.endswith(".tif")):
        continue

    # Load the image for inference
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image at path {image_path}. Skipping.")
        continue

    # Perform inference on the image
    outputs = predictor(image)

    # Visualize the results
    v = Visualizer(image[:, :, ::-1], scale=1.0)  # Convert BGR to RGB for visualization
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Save the output image with bounding boxes/masks
    output_image_name = os.path.splitext(image_name)[0] + "_inference.jpg"
    output_path = os.path.join(output_folder_path, output_image_name)
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

    print(f"Inference completed for {image_name}. Result saved to: {output_path}")
