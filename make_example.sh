nvcc -I /usr/local/cub-1.6.4/ -I /usr/local/cub-1.6.4/test/ -o example example.cu nested_loop.cu -lm --relocatable-device-code true
