#!/bin/bash
echo "Pipeline config path+name: "+$1
echo "Model path+name: "+$2
echo "Model output path: "+$3
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=$1 --trained_checkpoint_prefix=$2 --output_directory=$3