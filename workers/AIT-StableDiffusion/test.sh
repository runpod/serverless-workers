con_id=$(docker create merrell/ait-sd-2-runpod:latest)
docker cp ./test_input.json $con_id:/AITemplate/test_input.json
docker commit $con_id merrell/ait-sd-2-runpod:latest
docker rm -v $con_id

docker run --gpus all merrell/ait-sd-2-runpod:latest bash -c "cd AITemplate; python3 -u examples/05_stable_diffusion/infer.py"
con_id=$(docker ps -lq)
docker cp $con_id:/AITemplate/simulated_uploaded .
docker rm -v $con_id

docker run merrell/ait-sd-2-runpod:latest rm /AITemplate/test_input.json
con2_id=$(docker ps -lq)
docker commit $con2_id merrell/ait-sd-2-runpod:latest
docker rm -v $con2_id