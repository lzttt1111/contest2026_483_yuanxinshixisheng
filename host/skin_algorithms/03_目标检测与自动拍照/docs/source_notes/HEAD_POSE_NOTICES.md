# 第三方来源说明

最终 ONNX 来自 Omar Hassan 的 MIT 许可项目：

- 项目：https://github.com/ohtlab/headpose-fsanet-pytorch
- 文件：`pretrained/fsanet-1x1-iter-688590.onnx`
- 上游 SHA256：`d85e90cb4b9d3a87e81bbfdcfcd130cbe0e175407042e439a570aa396c44eda6`
- 许可证全文：`third_party_licenses/FSA_NET_PYTORCH_MIT.txt`

该实现基于 CVPR 2019 的 FSA-Net，原始实现位于 https://github.com/shamangary/FSA-Net（Apache-2.0）。本项目没有打包上游训练数据。

自动人脸定位使用 OpenCV Zoo 提供的 YuNet 2023mar ONNX：

- 项目：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- 文件：`face_detection_yunet_2023mar.onnx`
- SHA256：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- 许可证：MIT，全文见 `third_party_licenses/YUNET_MIT.txt`

上游 README 明确提示 300W-LP、AFLW2000、LFPW、HELEN、AFW、IBUG 等数据集各有自己的许可。使用预训练权重进行商业发布前，应由产品方完成数据集来源和适用许可的合规复核；本说明不是法律意见。
