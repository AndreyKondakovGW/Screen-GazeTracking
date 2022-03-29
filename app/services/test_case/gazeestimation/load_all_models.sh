#!/bin/bash

downloader_path=/home/godizigel/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py
optimizer_path=/home/godizigel/intel/openvino_2021/deployment_tools/model_optimizer/mo.py
if [ -f $downloader_path ]; then
    if ! [ -f ./modles/face-detection ]; then
        echo "Load face-detection model..."
        python3 $downloader_path --name face-detection-adas-0001 --output_dir ./models/face-detection --precisions FP32
    fi

    if ! [ -f ./modles/head-pose-estimation ]; then
        echo "Load head-pose-estimation model..."
        python3 $downloader_path --name head-pose-estimation-adas-0001 --output_dir ./models/head-pose-estimation  --precisions FP32
    fi

    if ! [ -f ./modles/facial-landmarks ]; then
        echo "Load facial-landmarks model..."
        python3 $downloader_path --name facial-landmarks-35-adas-0002 --output_dir ./models/facial-landmarks --precisions FP32
    fi

    if ! [ -f ./modles/open-closed-eye ]; then
        echo "Load open-closed-eye model..."
        python3 $downloader_path --name open-closed-eye-0001 --output_dir ./models/open-closed-eye --precisions FP32
        echo "Convert open-closed-eye model to IR..."
        python3 $optimizer_path --input_model ./models/open-closed-eye/public/open-closed-eye-0001/open-closed-eye.onnx --data_type FP32 --output_dir ./models/open-closed-eye-ir
    fi

    if ! [ -f ./modles/gaze-estimation ]; then
        echo "Load gaze-estimation model..."
        python3 $downloader_path --name gaze-estimation-adas-0002 --output_dir ./models/gaze-estimation --precisions FP32
    fi
else
    echo "wrond path $downloader_path, install openvion or check path to models downloader"
fi
