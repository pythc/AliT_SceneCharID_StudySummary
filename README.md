# Table Tennis Image Recognition Based on PP-YOLOE-SOD in Baidu PaddlePaddle Framework  

**Project Overview**  
This project focuses on table tennis image recognition using the PP-YOLOE-SOD model within the Baidu PaddlePaddle framework. It aims to detect table tennis balls and analyze bounce events in table tennis matches through computer vision techniques.  


## Project Background  
Table tennis is a fast-paced sport requiring precise ball tracking for training analysis, match replay, and intelligent referee systems. Traditional manual analysis is time-consuming and error-prone. This project leverages deep learning to automatically recognize table tennis balls in images/videos, providing technical support for sports big data analysis and intelligent sports applications.  


## Key Technologies  
- **PP-YOLOE-SOD**: A lightweight object detection model optimized for small objects, suitable for table tennis ball detection (small target in complex backgrounds).  
- **PaddlePaddle**: Baidu's open-source deep learning framework, supporting efficient model training and deployment.  
- **Data Augmentation**: Techniques like random cropping, color jittering, and horizontal flipping to improve model generalization.  


## Key Links  
- **Competition Link**: [第十六届中国大学生服务外包创新创业大赛](https://aistudio.baidu.com/competition/detail/1273/0/introduction)  
- **PaddlePaddle Framework**: [PaddleDetection GitHub Repository](https://github.com/PaddlePaddle/PaddleDetection)  
- **PP-YOLOE-SOD Model**: [Configuration Files](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1/configs/smalldet)  
- **Datasets**:  
  - [Table Tennis Dataset Part 1](https://aistudio.baidu.com/datasetdetail/321050)  
  - [Table Tennis Dataset Part 2](https://aistudio.baidu.com/datasetdetail/321057)  


## Environment Versions  
- **PaddlePaddle**: 2.6.2  
- **Python**: 3.10.10  
- **PaddleDetection**: 2.6.1
- **PaddleVideo 2.1.0
- **Platform**: Baidu AI Studio

## Usage Guide  
1. **Download Datasets**:  
   Download `train_part1.zip` and `train_part2.rar` from the provided links and extract them.  

2. **Configure File Paths**:  
   Update all file paths in the `main.py` file to match the local storage paths of the dataset and model files.  

3. **Run the Program**:  
   Execute the `main.py` script directly in the Baidu AI Studio environment to start image recognition and analysis.


## Contribution  
We welcome contributions! Please follow these steps:  
1. Fork the repository and create a new branch (`git checkout -b feature/new-module`).  
2. Make changes and ensure code compliance (formatting, comments).  
3. Submit a pull request (PR) with a clear description of the updates.  


## License  
This project is licensed under the [Apache License 2.0](LICENSE).  


## Contact  
For issues or collaboration inquiries, please open an [Issue]([https://github.com/pythc/Table-Tennis-Recognition-Basedon-PP-YOLOE-SOD/issues]) or email: `2813994715@qq.com`.  
