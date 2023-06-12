Lidar simulation package with Carla
-----------------------------------

This python package allows the generation of a large amount of synthetic lidar data using the CARLA engine
(based on Unreal Engine 4). The simulated realistic world is executed inside docker because it is impossible / very hard
to install CARLA on ubuntu 20.04.

lidar_carla_sim package
***********************

Make sure you have the latest `pip` version::

  $ pip install -U pip

Install the package using::

  $ pip install wheel.whl

Should work right out of the box.

nbstripout
**********

Please install `nbstripout` package and use::

  $ nbstripout --install

in source tree root if you ever push notebook modifications.


Building carla docker image
***************************

If you need to recreate the docker image follow these instructions. One first need to build the
unreal engine docker image using the ue4-docker python tool.

ue4-docker
==========

This is a python utility to build the Unreal Engine in a docker image. The
command I used uses the custom fork for Carla simulator. The command is::

  $ ue4-docker build --ue-version=custom -repo https://github.com/CarlaUnreal/UnrealEngine.git -branch carla -username USERNAME -password GITHUBTOKEN -v --cuda=10.2 --target full

This command takes a very long time to execute and thus should be used
with `nohup`::

  $ nohup cmd > cmd.log 2>&1 &
  $ echo $! > cmd.pid

There will be many images created but the useful one is the 'full': `adamrehn/ue4-full:custom`.
From this image we build the carla simulator with a Dockerfile.

BTW: I had to fix some bugs (chown bugs for some reasons) in this repo thus you should use
my fork instead of the one directly from pypi.

Carla with Docker
=================

Use the Dockerfile in this repo to build carla. We build it in a docker image
because only old `clang-8` compilers are supported and these are deprecated
on ubuntu 20. To build carla use the following command::

  $ docker build -t IMAGETAG . -f Dockerfile.carla

Again, this command takes a long time because shaders will be compiled at the same time
and it takes ~5hrs at least (with 12 core machine + gpu). Thus the `nohup` command
could be practical.

Again, I had to fix some bugs in the original carla build system and so you might want to
use my fork of carla (but perhaps it is not needed?!?).

If using another computer to create the image, you might want these commands::

  $ docker save IMAGETAG -o TARBALLNAME.tar
  $ gzip TARBALLNAME.tar
  $ exit
  $ scp REMOTE:PATH/TARBALLNAME.tar.gz .
  $ docker load -i TARBALLNAME.tar.gz


Running Carla Server
====================

To run the server stored in the docker image use this command::

  # This enables the GUI (if needed) to be shared from the local machine to the docker container
  $ xhost local:root
  $ docker run --privileged --user=ue4 --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=:1 -e SDL_VIDEODRIVER=x11 fgoud/lidar-in-the-loop-hyperparameter-optimization:latest /bin/bash Dist/CARLA_Shipping_9071d25-dirty/LinuxNoEditor/CarlaUE4.sh

However, the GUI can be very slow to use and thus simulations are better off without it. TO disable it just add the `-RenderOffScreen` (for the CarlaUE4.sh script).

Python Wheel
============

The python API for carla is built into a wheel that must be extracted out of the docker image. First run the container (like above)
then::

  $ docker cp CONTAINER_NAME:/home/ue4/carla/PythonAPI/carla/dist/carla-0.9.13-cp38-cp38-linux_x86_64.whl .
  $ (in venv) pip install ./carla-0.9.13-cp38-cp38-linux_x86_64.whl
