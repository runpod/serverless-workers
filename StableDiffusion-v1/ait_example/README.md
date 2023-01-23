1. I first used the default docker scripts given by docker to build the container (which is available here https://github.com/facebookincubator/AITemplate/tree/main/docker)

tag the container as merrell/ait-sd-1-runpod or merrell/ait-sd-2-runpod

note : the docker script does not build the stable diffusion images, those have to be built separately

2. I then wrote a compile script to be able to build the docker container via a shell script, this script is available under the AITemplate_docker repository in the vm justin provided me with,

I've created 3 scripts called compile.sh, compile-v1.5.sh and compile-anything.sh that should be able to compile and generate stable diffusion v2, v1.5 and anything v3

3. additionally, there are test scripts, test.sh, test-v1.5.sh and test-anything.sh that should allow for testing the docker scripts for generating images, the generated images should be visible in a folder called simulated_upload over there

4. you can modify the files and call the update.sh or test.sh scripts and it'll automatically modify the containers to have the updated sd scripts

building the stable diffusion from scratch takes about 10 - 15 minutes (or longer), and I'd recommend using the scripts I created for updating the files in the container
