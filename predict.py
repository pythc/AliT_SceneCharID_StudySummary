import sys
import os
import cv2
import json
import numpy as np
import paddle

def create_predictor(model_dir):
    """
    根据 model_dir 中的模型文件(model.pdmodel和model.pdiparams)创建 Paddle Inference Predictor.
    """
    model_path = os.path.join(model_dir, "model.pdmodel")
    params_path = os.path.join(model_dir, "model.pdiparams")
    config = paddle.inference.Config(model_path, params_path)
    # 配置使用GPU（如果有GPU）; 否则可用CPU
    if paddle.device.is_compiled_with_cuda():
        config.enable_use_gpu(100, 0)  # 分配100MB显存，设备ID 0
    else:
        config.disable_gpu()
    # 可以打开IR优化/十字交叉优化，提高推理速度
    config.switch_ir_optim(True)
    # 创建预测器
    predictor = paddle.inference.create_predictor(config)
    return predictor, config

def preprocess(image, target_size=(640, 640)):
    """
    对输入图片进行预处理：
      - 调整尺寸
      - 将 BGR 转换为 RGB（若模型训练时要求）
      - 转换为 float32 并归一化到 [0,1]
      - HWC 转 CHW，并增加 batch 维度
    """
    # 调整图片尺寸
    resized = cv2.resize(image, target_size)
    # 转换为 float32，并归一化（这里假设没有额外归一化，mean=[0,0,0], std=[1,1,1]）
    im = resized.astype('float32') / 255.0
    # 若需要，可以从 BGR 转 RGB，视训练预处理而定（常见情况：PaddleDetection中使用的是RGB）
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # HWC -> CHW
    im = np.transpose(im, (2, 0, 1))
    # 增加 batch 维度
    im = np.expand_dims(im, axis=0)
    return im, resized

def postprocess(output, score_threshold=0.25):
    """
    简单后处理：假设模型输出的预测结果格式为 shape=[N,6]，
    每行为 [cls_id, score, x1, y1, x2, y2]，过滤低置信度检测。
    返回过滤后的列表（每个检测为字典）。
    """
    pred = output.copy()  # numpy 数组
    results = []
    for det in pred:
        cls_id, score, x1, y1, x2, y2 = det
        if score < score_threshold:
            continue
        result = {
            "label": int(cls_id),
            "score": float(score),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        }
        results.append(result)
    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py data.txt result.json")
        sys.exit(1)

    data_file = sys.argv[1]
    result_file = sys.argv[2]

    # 假设 data.txt 中每一行是一个待推理图片的路径（可以是绝对或相对路径）
    with open(data_file, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    # 创建推理 Predictor，从 submission/model 目录下读取模型文件
    # 注意：这里假设提交包中模型文件放在 "model" 目录内
    predictor, config = create_predictor("model")

    # 获取输入、输出 tensor 的句柄
    input_names = predictor.get_input_names()  # 例如 ["image"]
    # 假设模型只有一个输入tensor
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()  # 例如 ["save_infer_model/scale_0.tmp_0"]
    output_tensor = predictor.get_output_handle(output_names[0])

    # 设置预处理、后处理参数（如 target_size 可根据训练设置而定）
    target_size = (640, 640)
    score_threshold = 0.25

    results_dict = {}

    total_time = 0.0
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue

        # 预处理：得到经过归一化处理的图像 tensor 数据
        im, resized = preprocess(image, target_size)
        # 设置输入（im shape: [1,3,H,W]）
        input_tensor.copy_from_cpu(im)
        # 推理
        start = paddle.time.time()
        predictor.run()
        end = paddle.time.time()
        total_time += (end - start)
        # 获取输出，假设输出为 numpy 数组形状 [N,6]
        output_data = output_tensor.copy_to_cpu()
        # 后处理：过滤低于阈值的检测，返回检测结果列表
        preds = postprocess(output_data, score_threshold)
        # 使用图片文件名作为 key（或者你也可以提取图片 id）
        results_dict[os.path.basename(img_path)] = preds

    # 将所有预测结果写入 result.json 文件
    with open(result_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # 打印总体推理速度信息
    if len(image_paths) > 0:
        avg_time = total_time / len(image_paths)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"Processed {len(image_paths)} images, average inference time: {avg_time:.4f}s, FPS: {fps:.2f}")

if __name__ == '__main__':
    main()
