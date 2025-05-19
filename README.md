
# Table Tennis Image Recognition Based on PP-YOLOE-SOD in Baidu PaddlePaddle Framework  

**Project Overview**  
This project focuses on table tennis image recognition using the PP-YOLOE-SOD model within the Baidu PaddlePaddle framework. It aims to detect table tennis balls and analyze bounce events in table tennis matches through computer vision techniques.  


### Key Links  
- **Competition Link**: [第十六届中国大学生服务外包创新创业大赛](https://aistudio.baidu.com/competition/detail/1273/0/introduction)  
- **PaddlePaddle Framework**: [PaddleDetection GitHub Repository](https://github.com/PaddlePaddle/PaddleDetection)  
- **PP-YOLOE-SOD Model**: [Configuration Files](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1/configs/smalldet)  
- **Datasets**:  
  - [Table Tennis Dataset Part 1](https://aistudio.baidu.com/datasetdetail/321050)  
  - [Table Tennis Dataset Part 2](https://aistudio.baidu.com/datasetdetail/321057)  


### Environment Versions  
- **PaddlePaddle**: 2.6.2  
- **Python**: 3.10.10  
- **PaddleDetection**: 2.6.1  
- **Platform**: Baidu AI Studio  


### Usage Guide  
1. **Download Datasets**:  
   Download `train_part1.zip` and `train_part2.rar` from the provided links and extract them.  

2. **Configure File Paths**:  
   Update all file paths in the `main.py` file to match the local storage paths of the dataset and model files.  

3. **Run the Program**:  
   Execute the `main.py` script directly in the Baidu AI Studio environment to start image recognition and analysis.  


This project leverages PaddlePaddle's deep learning capabilities and the PP-YOLOE-SOD model for small object detection to achieve accurate recognition of table tennis balls and bounce events, providing a technical solution for intelligent sports analysis.
