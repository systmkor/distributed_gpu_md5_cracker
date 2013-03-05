/* Authors: Luke Larson,
 *          Austin Munsch,
 *          Orion Miller
 *
 * Description: Milestone 2 
 *              See kernel.
 * Notes: This file will not compile.
 */

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

/* http://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda */
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

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

/* kernel */
__global__ void cudaCrack(char *hash, Word *words, int num_words, char *result) {
   int t = threadIdx.x + blockIdx.x * blockDim.x;
   md5_state_t state;
   md5_byte_t digest[16];
   char hex_output[16*2 + 1];
   int i;

   /* Iterate when more words than max threads */
   while (t < num_words) {
       md5_init(&state);
       md5_append(&state, (const md5_byte_t *) words[t].word, words[t].len);
       md5_finish(&state, digest);

       for (i = 0; i < 16; i++)
          sprintf(hex_output + i * 2, "%02x", digest[i]);

       /* If match, result correct word */
       if (strncmp(hex_output, hash, 32) == 0) {
          snprintf(result, words[t].len + 1, "%s", words[t].word);
          result[words[t].len] = '\0';
       }

       /* Add width of grid in threads to go through all words */
       t += blockDim.x * gridDim.x;
   }
}

void crackHash(char* hash, int dict) {
   int i;
   Word* words;
   struct stat fileStat;
   char* dictFile;
   int numWords;
   int curWord;
   char result[255] = {0};
   char word[WORD_LEN];
   char *cu_hash;
   char *cu_result;
   Word *cu_words;
   char *cu_result;

   md5_state_t state;
   md5_byte_t digest[16];
   char hex_output[16*2 + 1];

   /* Map the dictionary file */
   fstat(dict, &fileStat);
   dictFile = (char*)mmap(NULL, fileStat.st_size, PROT_READ, MAP_SHARED, dict, 0);

   /* Count the number of words */
   numWords = countWords(dictFile, fileStat.st_size);
   words = (Word*)malloc(sizeof(Word) * numWords);

   /* Fill in the array of words */
   initWords(words, dictFile, numWords, fileStat.st_size);

   HANDLE_ERROR(cudaMalloc(&cu_words, sizeof(Word) * numWords));
   HANDLE_ERROR(cudaMalloc(&cu_hash, 32+1));
   HANDLE_ERROR(cudaMalloc(&cu_result, 255));

   HANDLE_ERROR(cudaMemcpy(cu_words, words, sizeof(Word) * numWords, cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(cu_hash, hash, 32+1, cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpy(cu_result, result, 255, cudaMemcpyHostToDevice));

   dim3 grid(GRID); 
   dim3 block(BLOCK);
   cudaCrack<<<grid, block>>>(cu_hash, cu_words, numWords, result);

   HANDLE_ERROR(cudaMemcpy(result, cu_result, 255, cudaMemcpyDeviceToHost));
   
   if (result[0]) {
       printf("%s -> %s\n", hash, result);
       return;
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
