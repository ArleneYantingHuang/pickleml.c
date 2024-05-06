#include <stdlib.h>

float *read_file(const char *filename, size_t size);

void forward_linear_layer(float *output, float *input, float *weight,
                          float *bias, size_t input_size, size_t output_size);

void forward_elu(float *output, float *input, size_t size);

void forward_softmax(float *output, float *input, size_t size);