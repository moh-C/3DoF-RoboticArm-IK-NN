docker run --rm -it \
  --name tensorflow-gpu-jupyter \
  --network host \
  --gpus all \
  -v $(pwd):/tf/workdir \
  -w /tf/workdir \
  tensorflow/tensorflow:2.10.1-gpu-jupyter