# VehicleDetection

<p align="center">
<img src="tf.png" width="50%" height="50%" alignment="center">
</p>
Vehicle detection based on Nvidia graphics. Project was made on Brno University of Technology as Bachelors thesis. Retrained SSD MobileNet V2, SSD Inception V2 and Faster R-CNN Inception V2. These models have been executed on the Nvidia Tegra TX2 and Nvidia RTX 2060.

Three type of models TensorFlow model, TensorRT model and ONNX model.

### Precision of trained models.
| Model/IoU | Faster R-CNN | SSD Inception V2 | SSD MobileNet V2 |
|---|---|---|---|
| IoU=0.50:0.95 all | 0.396 | 0.294 | 0.361 | 
| IoU=0.50 all | 0.514|0.433 | 0.547 |
| IoU=0.75 all | 0.464|0.352 | 0.420 |
| IoU=0.50:0.95 small | 0.257 | 0.126 | 0.223|
| IoU=0.50:0.95 medium | 0.466 | 0.376 | 0.440|
| IoU=0.50:0.95 large | 0.442 | 0.444 | 0.426|

### Performance of trained models on Nvidia Tegra TX2 and Nvidia RTX 2060
| Model/Device performance | Nvidia RTX 2060 [FPS] || Nvidia Tegra TX2 [FPS]||
|---|---|---|---|---|
|| Batch=1 | Batch=MAX | Batch=1 | Batch=MAX |
| Faster R-CNN | 31.98 | B3\# 40.66 | 4.19 | XXX |
| TensorRT Faster R-CNN | 31.42 | B3\# 41.12 | 4.22 | XXX |
| ONNX Faster R-CNN | 20.4 | B3\# 19.12 | 0.39 | XXX |
| SSD Inception V2 | 97.61 | B65\# 252.58 | 15.60 | B21\# 32.41|
| TensorRT SSD Inception V2 | 96.46 | B65\# 237.78 | 15.89 | B21\# 32.48|
| ONNX SSD Inception V2 | 95.12 | B65\# 156.22 | 4.37 | B21\# 4.79|
| SSD MobileNet V2 | 175.02 | B41\# 297.91 | 22.76 | B14\# 43.80|
| TensorRT SSD MobileNet V2 | 184.47 | B41\# 300.77 | 23.17 | B14\# 43.77|
| ONNX SSD MobileNet V2 | 126.29 | B41\# 154.21 | 6.06 | B14\# 7.26|
