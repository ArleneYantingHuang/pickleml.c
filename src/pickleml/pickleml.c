#include "pickleml.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

float *read_file(const char *filename, size_t size) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Error: Can't open file\n");
    return NULL;
  }
  float *data = (float *)malloc(size * sizeof(float));
  if (data == NULL) {
    printf("Error: Can't allocate memory\n");
    return NULL;
  }
  size_t actual_size = fread(data, sizeof(float), size, fp);
  if (actual_size != size) {
    printf("Error: File smaller than expected\n");
    return NULL;
  }
  fclose(fp);
  return data;
}

// Y = W@X + b, W: weight = input_size * output_size, b: bias = output_size
// X: input = input_size, Y: output = output_size
void forward_linear_layer(float *output, float *input, float *weight,
                          float *bias, size_t input_size, size_t output_size) {
  for (size_t i = 0; i < output_size; i++) {
    output[i] = bias[i];
    for (size_t j = 0; j < input_size; j++) {
      output[i] += weight[j * output_size + i] * input[j];
    }
  }
}

void forward_elu(float *output, float *input, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input[i] > 0 ? input[i] : exp(input[i]) - 1;
  }
}

void forward_softmax(float *output, float *input, size_t size) {
  float sum = 0;
  for (size_t i = 0; i < size; i++) {
    output[i] = exp(input[i]);
    sum += output[i];
  }
  for (size_t i = 0; i < size; i++) {
    output[i] /= sum;
  }
}