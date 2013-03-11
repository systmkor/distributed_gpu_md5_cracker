#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "md5.h"

#define WORD_LEN 512

typedef struct Word {
   char* word;
   int len;
} Word;

int validateDictionary(char* filename);
char* validateHash(char* hash);
void crackHash(char* hash, int dict);
int countWords(char* dictFile, int size);
void initWords(Word* words, char* dictFile, int numWords, int size);

int main(int argc, char* argv[]) {
   int dict; /* Dictionary file descriptor */
   char* hash;
   char hash2[33] = {0};
   md5_byte_t hash_digest[16] = {0};

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
   int i;
   Word* words;
   struct stat fileStat;
   char* dictFile;
   int numWords;
   int curWord;
   char word[WORD_LEN];

   md5_state_t state;
   md5_byte_t digest[16];
   char hex_output[16*2 + 1];

   /* Map the dictionary file */
   fstat(dict, &fileStat);
   dictFile = mmap(NULL, fileStat.st_size, PROT_READ, MAP_SHARED, dict, 0);

   /* Count the number of words */
   numWords = countWords(dictFile, fileStat.st_size);
   words = malloc(sizeof(Word) * numWords);

   /* Fill in the array of words */
   initWords(words, dictFile, numWords, fileStat.st_size);

   /* Hash word and compare */
   for (curWord = 0; curWord < numWords; curWord++) {
      /* Hash */
      md5_init(&state);
      md5_append(&state, (const md5_byte_t *) words[curWord].word, words[curWord].len);
      md5_finish(&state, digest);
      for (i = 0; i < 16; i++)
         sprintf(hex_output + i * 2, "%02x", digest[i]);

      /* Compare */
      if (strncmp(hex_output, hash, 16) == 0) {
         snprintf(word, words[curWord].len + 1, "%s", words[curWord].word);
         word[words[curWord].len] = '\0';
         printf("%s matches\n", word);
         return;
      }
   }

   printf("Tested against %d words, no match\n", numWords);
}

int countWords(char* dictFile, int size) {
   int count = 0;
   int i = 0;

   for (i = 0; i < size; i++) {
      if (dictFile[i] == '\n')
         count++;
   }

   return count;
}

void initWords(Word* words, char* dictFile, int numWords, int size) {
   int i = 0;
   int wordLen = 0;
   int curWord = 0;

   for (i = 0; i < size; i++) {
      if (dictFile[i] == '\n') {
         words[curWord].len = wordLen - 1;
         words[curWord].word = &(dictFile[i]) - wordLen + 1;
         curWord++;
         wordLen = 0;
      }
      wordLen++;
   }
}
