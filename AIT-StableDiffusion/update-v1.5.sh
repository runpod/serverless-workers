docker run merrell/ait-sd-1-runpod:latest rm /AITemplate/examples/05_stable_diffusion-v1.5/ -r
con2_id=$(docker ps -lq)
docker commit $con2_id merrell/ait-sd-1-runpod:latest
docker rm -v $con2_id

con_id=$(docker create merrell/ait-sd-1-runpod:latest)
docker cp ./05_stable_diffusion-v1.5 $con_id:/AITemplate/examples/
docker commit $con_id merrell/ait-sd-1-runpod:latest
docker rm -v $con_id