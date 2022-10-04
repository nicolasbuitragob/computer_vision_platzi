#!/bin/bash

docker build -t computer_vision_platzi .

docker run --rm -v $PWD/app:/app -it -p 8000:8000 computer_vision_platzi