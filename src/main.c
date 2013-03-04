#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "md5.h"

#define WORD_LEN 512

int validateDictionary(char* filename);
char* validateHash(char* hash);
void crackHash(char* hash, int dict);

int main(int argc, char* argv[]) {
   int dict; /* Dictionary file descriptor */
   char* hash;

   /* Validate arguments */
   if (argc < 3) {
      printf("Usage: md5cracker dictionary-file hash\n");
      return EXIT_FAILURE;
   }
   dict = validateDictionary(argv[1]);
   hash = validateHash(argv[2]);

   /* Attempt to crack the hash */
   crackHash(hash, dict);

   close(dict);

   return EXIT_SUCCESS;
}

int validateDictionary(char* filename) {
   int fd;
   fd = open(filename, O_RDONLY);

   if (fd == -1) {
      printf("Failed to open dictionary file\n");
      exit(EXIT_FAILURE);
   }

   return fd;
}

char* validateHash(char* hash) {
   int i;

   /* Hash must be 32 characters long */
   if (strlen(hash) != 32) {
      printf("Invalid hash\n");
      exit(EXIT_FAILURE);
   }

   
   /* Each character must be a hex character */
   for (i = 0; i < strlen(hash); i++) {
      hash[i] = tolower(hash[i]);
      if (!((hash[i] >= 'a' && hash[i] <= 'f') || (hash[i] >= '0' && hash[i] <= '9'))) {
         printf("Invalid hash\n");
         exit(EXIT_FAILURE);
      }
   }

   return hash;
}

void crackHash(char* hash, int dict) {
   char word[WORD_LEN + 1];
   char c;
   int i;
   int j = 0;
   int readReturn;
   char* words;

   md5_state_t state;
   md5_byte_t digest[16];
   char hex_output[16*2 + 1];

   while (1) {
      /* Read in 1 word from dictionary */
      i = 0;
      do {
         readReturn = read(dict, &c, 1);
         if (readReturn == -1) {
            printf("Failed to read from input file\n");
            exit(EXIT_FAILURE);
         }
         if (readReturn == 0) {
            c = EOF;
            break;
         }
         if (c == '\n' || c == EOF || i == WORD_LEN)
            break;

         word[i] = c;
         i++;
      } while (1);
      word[i] = '\0';

      /* End condition */
      if (c == EOF)
         break;

      /* Hash word and compare */
      md5_init(&state);
      md5_append(&state, (const md5_byte_t *) word, strlen(word));
      md5_finish(&state, digest);
      for (i = 0; i < 16; i++)
         sprintf(hex_output + i * 2, "%02x", digest[i]);
      if (strncmp(hex_output, hash, 16) == 0) {
         printf("%s matches\n", word);
         return;
      }

      j++;
   }

   printf("Tested against %d words, no match\n", j);
}
