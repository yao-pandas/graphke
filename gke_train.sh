g++ -c basic_settings.c -fopenmp
g++ -c graph_reader.c
g++ -c hash_graph.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fopenmp
g++ -c init_embeds.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -fopenmp
nvcc -rdc=true -c user_interface.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
nvcc -rdc=true -c gpu_score.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
g++ -c validate.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
nvcc -rdc=true -c gpu_sample.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
nvcc -rdc=true -c gpu_validate.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
nvcc -rdc=true -c loss.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
nvcc -dlink gpu_sample.o gpu_validate.o gpu_score.o user_interface.o loss.o -o link.o 
g++ -o gke_train gke_train.c link.o loss.o validate.o gpu_validate.o user_interface.o gpu_score.o gpu_sample.o hash_graph.o \
	graph_reader.o init_embeds.o basic_settings.o \
	-I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -fopenmp -lm -lopenblas
rm *.o
