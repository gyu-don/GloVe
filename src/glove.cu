//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#include <cuda_runtime.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int verbose = 2; // 0, 1, or 2
int use_unk_vec = 1; // 0 or 1
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 0; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
int checkpoint_every = 0; // checkpoint the model for every checkpoint_every iterations. Do nothing if checkpoint_every <= 0
real eta_cpu = 0.05; // Initial learning rate
real alpha_cpu = 0.75, x_max_cpu = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *W, *gradsq, *cost;
real *W_gpu, *gradsq_gpu;
unsigned int vocab_size, num_lines;
char *vocab_file, *input_file, *save_W_file, *save_gradsq_file;

__constant__ unsigned voca_size, num_lines_gpu;
__constant__ real eta, alpha, x_max;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
	long long a, b;
	vector_size++; // Temporarily increment to allocate space for bias
    
	/* Allocate space for word vectors and context word vectors, and correspodning gradsq */
	a = posix_memalign((void **)&W, 128, 2 * vocab_size * (vector_size + 1) * sizeof(real)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, 2 * vocab_size * (vector_size + 1) * sizeof(real)); // Might perform better than malloc
	if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        exit(1);
    }
    if(cudaMalloc((void**)&W_gpu, 2 * vocab_size * (vector_size + 1) * sizeof(real)) != cudaSuccess) {
        printf("W alloc error\n");
        exit(1);
    }
    if(cudaMalloc((void**)&gradsq_gpu, 2 * vocab_size * (vector_size + 1) * sizeof(real)) != cudaSuccess) {
        printf("gradsq alloc error\n");
        exit(1);
    }

	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) W[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
	for (b = 0; b < vector_size; b++) for (a = 0; a < 2 * vocab_size; a++) gradsq[a * vector_size + b] = 1.0; // So initial value of eta is equal to initial learning rate
    cudaMemcpy(W_gpu, W, 2 * vocab_size * (vector_size + 1) * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(gradsq_gpu, gradsq, 2 * vocab_size * (vector_size + 1) * sizeof(real), cudaMemcpyHostToDevice);
	vector_size--;
}

__host__ __device__ inline real check_nan(real update) {
    if (isnan(update) || isinf(update)) {
        printf("\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

/* Train the GloVe model */
__global__ void glove_thread(real* cost, const CREC* data, real* W, real* gradsq, int steps) {
    long long a, l1, l2;
    unsigned int id = blockIdx.x;
    unsigned int b = threadIdx.x;
    int vec_size = blockDim.x;
    real temp1, temp2;
    extern __shared__ real diff[];
    real fdiff;
    int data_idx = steps * id;
    int data_end = steps * (id + 1);
    int data_len;
    if (data_end > num_lines_gpu) data_end = num_lines_gpu;
    data_len = data_end - data_idx;
    cost[id] = 0;

    real W_updates1;
    real W_updates2;
    //printf("Thread: %d, data_idx: %d, steps: %d, data_len: %d\n", id, data_idx, steps, data_len);
    for (a = 0; a < data_len; a++) {
        const CREC &cr = data[data_idx++];
        if (cr.word1 < 1 || cr.word2 < 1) { continue; }

        /* Get location of words in W & gradsq */
        l1 = (cr.word1 - 1LL) * (vec_size + 1); // cr word indices start at 1
        l2 = ((cr.word2 - 1LL) + voca_size) * (vec_size + 1); // shift by vocab_size to get separate vectors for context words

        /* Calculate cost, save diff for gradients */
        diff[b] = W[b + l1] * W[b + l2]; // dot product of word and context word vector
        __syncthreads();
        // 0x1000はスレッド数の半分よりも大きい数。
        for(unsigned int s = 0x1000;s>0;s>>=1) {
            if(b < s && b + s < vec_size) diff[b] = diff[b] + diff[b + s];
            __syncthreads();
        }
        if (b == 0) diff[0] += W[vec_size + l1] + W[vec_size + l2] - log(cr.val);
        __syncthreads();
        fdiff = (cr.val > x_max) ? diff[0] : pow(cr.val / x_max, alpha) * diff[0]; // multiply weighting function (f) with diff

        // Check for NaN and inf() in the diffs.
        if (isnan(diff[0]) || isnan(fdiff) || isinf(diff[0]) || isinf(fdiff)) {
            printf("Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }

        if (b == 0) cost[id] += 0.5 * fdiff * diff[0]; // weighted squared error
        
        /* Adaptive gradient updates */
        fdiff *= eta; // for ease in calculating gradient

        // learning rate times gradient for word vectors
        temp1 = fdiff * W[b + l2];
        temp2 = fdiff * W[b + l1];
        // adaptive updates
        W_updates1 = temp1 / sqrt(gradsq[b + l1]);
        W_updates2 = temp2 / sqrt(gradsq[b + l2]);
        gradsq[b + l1] += temp1 * temp1;
        gradsq[b + l2] += temp2 * temp2;
            
        if (!isnan(W_updates1) && !isinf(W_updates1) && !isnan(W_updates2) && !isinf(W_updates2)) {
            W[b + l1] -= W_updates1;
            W[b + l2] -= W_updates2;
        }

        // updates for bias terms
        if (b==0) {
            W[vec_size + l1] -= check_nan(fdiff / sqrt(gradsq[vec_size + l1]));
            W[vec_size + l2] -= check_nan(fdiff / sqrt(gradsq[vec_size + l2]));
            fdiff *= fdiff;
            gradsq[vec_size + l1] += fdiff;
            gradsq[vec_size + l2] += fdiff;
        }
        __syncthreads();
    }
}

/* Save params to file */
int save_params(int nb_iter) {
    /*
     * nb_iter is the number of iteration (= a full pass through the cooccurrence matrix).
     *   nb_iter > 0 => checkpointing the intermediate parameters, so nb_iter is in the filename of output file.
     *   else        => saving the final paramters, so nb_iter is ignored.
     */

    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH], output_file_gsq[MAX_STRING_LENGTH];
    char *word = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH + 1);
    FILE *fid, *fout, *fgs;
    
    if (use_binary > 0) { // Save parameters in binary file
        if (nb_iter <= 0)
            sprintf(output_file,"%s.bin",save_W_file);
        else
            sprintf(output_file,"%s.%03d.bin",save_W_file,nb_iter);

        fout = fopen(output_file,"wb");
        if (fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        for (a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&W[a], sizeof(real), 1,fout);
        fclose(fout);
        if (save_gradsq > 0) {
            if (nb_iter <= 0)
                sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            else
                sprintf(output_file_gsq,"%s.%03d.bin",save_gradsq_file,nb_iter);

            fgs = fopen(output_file_gsq,"wb");
            if (fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
            for (a = 0; a < 2 * (long long)vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(real), 1,fgs);
            fclose(fgs);
        }
    }
    if (use_binary != 1) { // Save parameters in text file
        if (nb_iter <= 0)
            sprintf(output_file,"%s.txt",save_W_file);
        else
            sprintf(output_file,"%s.%03d.txt",save_W_file,nb_iter);
        if (save_gradsq > 0) {
            if (nb_iter <= 0)
                sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            else
                sprintf(output_file_gsq,"%s.%03d.txt",save_gradsq_file,nb_iter);

            fgs = fopen(output_file_gsq,"wb");
            if (fgs == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_gradsq_file); return 1;}
        }
        fout = fopen(output_file,"wb");
        if (fout == NULL) {fprintf(stderr, "Unable to open file %s.\n",save_W_file); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if (fid == NULL) {fprintf(stderr, "Unable to open file %s.\n",vocab_file); return 1;}
        for (a = 0; a < vocab_size; a++) {
            if (fscanf(fid,format,word) == 0) return 1;
            // input vocab cannot contain special <unk> keyword
            if (strcmp(word, "<unk>") == 0) return 1;
            fprintf(fout, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if (save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if (fscanf(fid,format,word) == 0) return 1; // Eat irrelevant frequency entry
        }

        if (use_unk_vec) {
            real* unk_vec = (real*)calloc((vector_size + 1), sizeof(real));
            real* unk_context = (real*)calloc((vector_size + 1), sizeof(real));
            word = "<unk>";

            int num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for (a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for (b = 0; b < (vector_size + 1); b++) {
                    unk_vec[b] += W[a * (vector_size + 1) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + 1) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_vec[b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            fprintf(fout,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fout);
        if (save_gradsq > 0) fclose(fgs);
    }
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size;
    int save_params_return_code;
    int b;
    FILE *fin;
    real total_cost = 0;
    CREC *data_host, *data;
    int steps;
    real *cost_gpu;


    fprintf(stderr, "TRAINING MODEL\n");
    if (cudaMalloc(&cost_gpu, num_threads * sizeof(real)) != cudaSuccess) {
        printf("allocation failed.\n");
        exit(1);
    }
    
    fin = fopen(input_file, "rb");
    if (fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    data_host = (CREC*)malloc(file_size);
    if (!data_host) {
        printf("host allocation failed.\n");
        exit(1);
    }
    fseeko(fin, 0, SEEK_SET);
    fread(data_host, sizeof(CREC), num_lines, fin);
    fclose(fin);
    if (cudaMalloc(&data, file_size) != cudaSuccess) {
        printf("allocation failed.\n");
        exit(1);
    }
    if (cudaMemcpy(data, data_host, file_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("memcpy failed.\n");
        exit(1);
    }
    free(data_host);

    if (cudaMemcpyToSymbol(num_lines_gpu, &num_lines, sizeof(num_lines)) != cudaSuccess) {
        printf("nu failed.\n");
        exit(1);
    }

    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if (verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if (verbose > 1) fprintf(stderr,"done.\n");
    if (verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if (verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max_cpu);
    if (verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha_cpu);
    
    time_t rawtime;
    struct tm *info;
    char time_buffer[80];
    steps = num_lines / num_threads;
    // Lock-free asynchronous SGD
    for (b = 0; b < num_iter; b++) {
        total_cost = 0;
        glove_thread<<<num_threads, vector_size, sizeof(real) * vector_size>>>(cost_gpu, data, W_gpu, gradsq_gpu, steps);
        cudaMemcpy(cost, cost_gpu, num_threads * sizeof(real), cudaMemcpyDeviceToHost);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer,80,"%x - %I:%M.%S%p", info);
        fprintf(stderr, "%s, iter: %03d, cost: %lf\n", time_buffer,  b+1, total_cost/num_lines);

        if (checkpoint_every > 0 && (b + 1) % checkpoint_every == 0) {
            fprintf(stderr,"    saving itermediate parameters for iter %03d...", b+1);
            save_params_return_code = save_params(b+1);
            if (save_params_return_code != 0)
                return save_params_return_code;
            fprintf(stderr,"done.\n");
        }

    }
    if(cudaMemcpy(W, W_gpu, 2 * vocab_size * (vector_size + 2) * sizeof(real), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("memcpy err\n");
        exit(1);
    }
    if(cudaMemcpy(gradsq, gradsq_gpu, 2 * vocab_size * (vector_size + 2) * sizeof(real), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("memcpy err\n");
        exit(1);
    }
    cudaFree(cost_gpu);
    cudaFree(data);
    return save_params(0);
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    vocab_file = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH);
    input_file = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_W_file = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH);
    save_gradsq_file = (char*)malloc(sizeof(char) * MAX_STRING_LENGTH);
    int result = 0;
    
    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\t-checkpoint-every <int>\n");
        printf("\t\tCheckpoint a  model every <int> iterations; default 0 (off)\n");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        result = 0;
    } else {
        if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        cost = (real*)malloc(sizeof(real) * num_threads);
        if ((i = find_arg((char *)"-alpha", argc, argv)) > 0) alpha_cpu = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-x-max", argc, argv)) > 0) x_max_cpu = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta_cpu = atof(argv[i + 1]);
        if ((i = find_arg((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
        if (model != 0 && model != 1) model = 2;
        if ((i = find_arg((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
        if ((i = find_arg((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
        else strcpy(vocab_file, (char *)"vocab.txt");
        if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
        else strcpy(save_W_file, (char *)"vectors");
        if ((i = find_arg((char *)"-gradsq-file", argc, argv)) > 0) {
            strcpy(save_gradsq_file, argv[i + 1]);
            save_gradsq = 1;
        }
        else if (save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
        if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
        else strcpy(input_file, (char *)"cooccurrence.shuf.bin");
        if ((i = find_arg((char *)"-checkpoint-every", argc, argv)) > 0) checkpoint_every = atoi(argv[i + 1]);
        
        vocab_size = 0;
        fid = fopen(vocab_file, "r");
        if (fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n",vocab_file); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
        fclose(fid);

        if(cudaMemcpyToSymbol(voca_size, &vocab_size, sizeof(vocab_size)) != cudaSuccess) {
            printf("vo fail\n"); exit(1);
        }
        if(cudaMemcpyToSymbol(alpha, &alpha_cpu, sizeof(alpha_cpu)) != cudaSuccess) {
            printf("al fail\n"); exit(1);
        }
        if(cudaMemcpyToSymbol(x_max, &x_max_cpu, sizeof(x_max_cpu)) != cudaSuccess) {
            printf("xm fail\n"); exit(1);
        }
        if(cudaMemcpyToSymbol(eta, &eta_cpu, sizeof(eta_cpu)) != cudaSuccess) {
            printf("et fail\n"); exit(1);
        }
        result = train_glove();
        free(cost);
    }
    free(vocab_file);
    free(input_file);
    free(save_W_file);
    free(save_gradsq_file);
    return result;
}
