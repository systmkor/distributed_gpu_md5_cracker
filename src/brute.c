// Luke Larson
// a simple brute force algorithm over
// a predefined character set.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789"; 
    char buffer[255] = {0};
    int int_buffer[255] = {0};
    int chars_length, str_length = 6;
    int i,j=0;

    chars_length = strlen(chars);
    memset(buffer, chars[0], str_length);

    while (1) {
        if (j++ % 1000000 == 0)
            printf("%s\n", buffer);
        int_buffer[0]++;
        for (i = 0; i < str_length; i++) {
            if (int_buffer[i] == chars_length) {
                if (i+1 == str_length)
                    goto out;
                int_buffer[i] = 0;
                buffer[i] = chars[0];
                int_buffer[i+1]++;
            }
            else {
                buffer[i] = chars[int_buffer[i]];
                break;
            }
        }
    }
out:

    return 0;
}
