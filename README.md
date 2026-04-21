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

From now on,we call geometry_analyzer as GA.
GA calculate the diameter and area of　obgect from segmentation.
There are various ways to define the diameter, but in this study, the direction is determined using PCA, and the Feret diameter along that direction is adopted.
This is because it is close to the size that humans perceive as the diameter when observing an object.

Those calcilated value is saved in a file, so you can plot it by Excel or something.
<img width="362" height="218" alt="image" src="https://github.com/user-attachments/assets/1f1b8284-40c3-4f8e-95a4-d314fcff98f5" />


*we can't the sourse code of Operando Master to ensure the protection of research information

--------------------------------------------

TEM画像の幾何情報を解析するためのツールです。
本プログラムは、Operando Masterのプラグインの一つです。

Operando Masterは、TEM画像を解析するためのシステムであり、主に名古屋大学の研究室に所属するTangによって開発されました。
このシステムでは、YOLOv5とU-Net++の2つのCNNモデルを使用しています。

Operando Masterは、3種類の画像を生成します。

入力：TEM画像

ステップ1：YOLOにより、学習済みの対象物を検出します。

ステップ2：U-Net++により、対象物のセグメンテーションを行います。

ステップ3：geometry_analyzerにより幾何情報を計算し、セグメンテーション画像に付加します。

<img width="556" height="530" alt="スクリーンショット 2026-04-21 17 05 22" src="https://github.com/user-attachments/assets/ec7a7312-e8f2-4154-928a-84b141935fef" />
以降、geometry_analyzerをGAと呼びます。
GAは、セグメンテーション画像から対象物の直径および面積を計算します。

直径の定義には様々な方法がありますが、本研究ではPCAによって方向を決定し、その方向におけるフェレ径を採用しています。
これは、人間が物体を観察した際に直径として認識するサイズに近いためです。

また計算された値は別のファイルに保存されます。エクセルなどのツールでその値をプロットすることも可能です。
<img width="362" height="218" alt="image" src="https://github.com/user-attachments/assets/1f1b8284-40c3-4f8e-95a4-d314fcff98f5" />

なお研究情報の保護のためOperando Masterを公開することはできません。私が開発したGAはその一部で機能するプラグインであると認識してください。
