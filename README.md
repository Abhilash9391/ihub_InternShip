Second Assignement : 
 Object Detection and segmentation on multiple images in a single program 
<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/ba517e62-7d1c-4d04-832a-928b22d739eb" />
<img width="2250" height="1500" alt="image" src="https://github.com/user-attachments/assets/06775202-47c0-4b2a-9dbb-c5c7359a3d82" />

Used Coco128 to train the model Yolo128n.pt for just 3 epochs 


AsL SignLanguage to text converter

Creating Dataset :
=>from 5 youtube videos taken out the ASL signs from 5 different people 
=>and using makesense.ai(as i cannot download label-studio due to network issues) labelled bounding boxes for hands 
=>exported as zip 
=>using albumentations performed Data Augumentation on both iamges and bounding boxes at the same time

Model Training :
The system employs a YOLOv8 object detection model, fine-tuned through transfer learning for real-time gesture detection and classification.

Results:

<img width="1920" height="1648" alt="image" src="https://github.com/user-attachments/assets/c21741c9-c41e-48cd-b944-b2c3b5192077" />
<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/0a820ba4-ccd0-4706-a4d3-182f1c039f98" />
<img width="1600" height="1600" alt="image" src="https://github.com/user-attachments/assets/dfcc2a55-8233-484b-9f11-035880feb458" />

<img width="3000" height="2250" alt="image" src="https://github.com/user-attachments/assets/77ee97dc-19d8-441c-87f3-279a51323d34" />



