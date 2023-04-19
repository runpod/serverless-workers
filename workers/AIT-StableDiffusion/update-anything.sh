docker run merrell/ait-sd-1-runpod:anything rm /AITemplate/examples/05_stable_diffusion-anything-v3/ -r
con2_id=$(docker ps -lq)
docker commit $con2_id merrell/ait-sd-1-runpod:anything
docker rm -v $con2_id

con_id=$(docker create merrell/ait-sd-1-runpod:anything)
docker cp ./05_stable_diffusion-anything-v3/ $con_id:/AITemplate/examples/
docker commit $con_id merrell/ait-sd-1-runpod:anything
docker rm -v $con_id