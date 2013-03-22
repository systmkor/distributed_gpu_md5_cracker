all:
	nvcc -O3 -arch=sm_20 md5gcracker.cu -o md5gcracker
	gcc -O3 md5cracker.c md5.c -o md5cracker
	gcc -O3 passgen.c -o passgen 

clean:
	rm -f md5gcracker md5cracker passgen
