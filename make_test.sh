nvcc -I /usr/local/cub-1.6.4/ -I /usr/local/cub-1.6.4/test/ -o test_nested_loop test_nested_loop.cu nested_loop.cu -lm --relocatable-device-code true
