#include <time.h>
#include <stdio.h>
#include <stdlib.h>
 
/* Based on: http://en.wikipedia.org/wiki/Random_password_generator */
int main(int argc, char *argv[])
{
    /* Length of the password */
    unsigned short int length;
    unsigned long i;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <numwords>\n", argv[0]);
        return 1;
    }

    if (!atoi(argv[1])) {
        fprintf(stderr, "Bad argument\n");
        return 1;
    }

    /* Seed number for rand() */
    srand((unsigned int) time(0) + getpid());

    for (i = 0; i < atoi(argv[1]); i++) {
        length = 8;
        /* ASCII characters 33 to 126 */
        while(length--) 
            printf("%c", rand() % 94 + 33);
        printf("\n");
    }

    return EXIT_SUCCESS;
}
