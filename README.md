# geometry_analyzer
To analyze some gemetry infomations of TEM images.
This is one of the plugins of Operando Master.
Operando Master is the one and only system to study TEM images and mainly developed by Tang who take pert in NagoyaUniv labratry.
It use two CNN models,YOLOv5 and U-Net++.

Operando Master produce 3 images.
Input:TEM images
Step1:YOLO serch the object you made learned
Step2:U-Net make segmentation of the object
Step3:geometry_analyzer calcurate gemetry infomations and add it into images of segmentation.
<img width="556" height="530" alt="スクリーンショット 2026-04-21 17 05 22" src="https://github.com/user-attachments/assets/ec7a7312-e8f2-4154-928a-84b141935fef" />
