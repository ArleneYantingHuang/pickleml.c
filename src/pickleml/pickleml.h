#include <stdint.h>
#include <stdlib.h>

typedef struct {
  float *weight;
  float *bias;
  int input_size;
  int output_size;
} LinearLayer;

float *read_file(const char *filename, size_t size);

void forward_linear_layer(float *output, float *input,
                          LinearLayer *linear_layer, size_t input_size,
                          size_t output_size);

void forward_elu(float *output, float *input, size_t size);

void forward_softmax(float *output, float *input, size_t size);

void image_normalize(uint8_t *image, float *normalized_image, size_t size,
                     float scale);

int arg_max(float *output, size_t size);