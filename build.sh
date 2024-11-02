#!/bin/bash -e

rm -rf out/
mkdir -p out/

gcc -S -std=c23 -g3 -DBUILD_DEBUG=1 -pedantic -Wall -Werror -Wextra -Wswitch-enum -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -march=native -I. -lm -lSDL2 src/main_sdl2.c -o out/bzsim_sdl2.asm

gcc -std=c23 -g3 -DBUILD_DEBUG=1 -pedantic -Wall -Werror -Wextra -Wswitch-enum -Wconversion -Wdouble-promotion -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion -march=native -I. -lm -lSDL2 src/main_sdl2.c -o out/bzsim_sdl2

out/bzsim_sdl2
