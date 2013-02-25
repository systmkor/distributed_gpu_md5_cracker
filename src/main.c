#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>

int validateDictionary(char* filename);
char* validateHash(char* hash);

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

   printf("%s\n", hash);

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
