#  CPE 458
#  Luke Larson
#  Orion Miller
#  Austin Munsch

CC=gcc
CFLAGS=-Wall -g -ansi -pedantic -D_XOPEN_SOURCE=500
LDFLAGS=

ALL=md5cracker

all:	$(ALL)

md5cracker:	md5.c main.c
		$(CC) $(CFLAGS) $(LDFLAGS) -o $@ md5.c main.c

clean:
		rm -rf core* *.o *.gch $(ALL)
