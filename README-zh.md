
# 基于百度飞桨PP-YOLOE-SOD的乒乓球图像识别  

**项目概述**  
本项目专注于基于百度飞桨（PaddlePaddle）框架，使用PP-YOLOE-SOD模型实现乒乓球图像识别，旨在通过计算机视觉技术检测乒乓球并分析乒乓球比赛中的弹起事件。  


## 项目背景  
乒乓球是一项快节奏的运动，训练分析、比赛回放和智能裁判系统均需要精准的球体追踪。传统人工分析耗时且容易出错，本项目利用深度学习技术自动识别图像/视频中的乒乓球，为体育大数据分析和智能体育应用提供技术支持。  


## 关键技术  
- **PP-YOLOE-SOD**：针对小目标优化的轻量级目标检测模型，适用于复杂背景下的乒乓球检测（球体在图像中通常为小目标）。  
- **PaddlePaddle**：百度开源深度学习框架，支持高效的模型训练与部署。  
- **数据增强**：采用随机裁剪、颜色抖动、水平翻转等技术提升模型泛化能力。  


## 关键链接  
- **比赛链接**：[第十六届中国大学生服务外包创新创业大赛](https://aistudio.baidu.com/competition/detail/1273/0/introduction)  
- **PaddlePaddle框架**：[PaddleDetection GitHub仓库](https://github.com/PaddlePaddle/PaddleDetection)  
- **PP-YOLOE-SOD模型**：[配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1/configs/smalldet)  
- **数据集**：  
  - [乒乓球数据集Part 1](https://aistudio.baidu.com/datasetdetail/321050)  
  - [乒乓球数据集Part 2](https://aistudio.baidu.com/datasetdetail/321057)  


## 环境版本  
- **PaddlePaddle**：2.6.2  
- **Python**：3.10.10  
- **PaddleDetection**：2.6.1  
- **PaddleVideo**：2.1.0  
- **平台**：百度AI Studio  


## 使用指南  
1. **下载数据集**：  
   从提供的链接下载`train_part1.zip`和`train_part2.rar`并解压。  

2. **配置文件路径**：  
   更新`main.py`文件中所有文件路径，使其与数据集和模型文件的本地存储路径一致。  

3. **运行程序**：  
   在百度AI Studio环境中直接执行`main.py`脚本，启动图像识别与分析。  


## 贡献指南  
欢迎贡献代码！请遵循以下步骤：  
1. Fork本仓库并创建新分支（`git checkout -b feature/新功能`）。  
2. 进行代码修改并确保符合格式规范（代码风格、注释等）。  
3. 提交Pull Request（PR）并清晰描述更新内容。  


## 许可证  
本项目采用[Apache 2.0许可证](LICENSE)开源。  


## 联系方式  
如有问题或合作意向，请创建[Issue](https://github.com/pythc/Table-Tennis-Recognition-Basedon-PP-YOLOE-SOD/issues)或发送邮件至：`2813994715@qq.com`。
