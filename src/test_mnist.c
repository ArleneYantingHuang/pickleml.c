#include "pickleml/pickleml.h"
#include "raylib.h"
#include "stdint.h"
#include <stdio.h>

typedef struct {
  float *weight;
  float *bias;
  int input_size;
  int output_size;
} LinearLayer;

typedef struct {
  LinearLayer linear0;
  LinearLayer linear2;
} TwoLayerNet;

int predict_digit(uint8_t *image, TwoLayerNet *model) {
  // TOOD: Implement neural network
  // print image
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      printf("%d ", image[i * 28 + j]);
    }
    printf("\n");
  }

  return 0;
}

int main(int argc, char *argv[]) {
  const int mnist_size = 28;
  const int pixel_size = 10;

  const int window_width = mnist_size * pixel_size;
  const int window_height = mnist_size * pixel_size;

  InitWindow(window_width, window_height, "MNIST Test");
  SetTargetFPS(1000);

  // Allocate mnist texture
  RenderTexture2D mnist_texture = LoadRenderTexture(mnist_size, mnist_size);
  // Initialize mnist image
  BeginTextureMode(mnist_texture);
  ClearBackground(GetColor(0x000000FF));
  EndTextureMode();

  // Load mnist model
  float *mnist_model_0_weight =
      read_file("./binary/mnist_model_0_weight.bin", 28 * 28 * 256);
  float *mnist_model_0_bias = read_file("./binary/mnist_model_0_bias.bin", 256);
  LinearLayer linear0 = {mnist_model_0_weight, mnist_model_0_bias, 28 * 28, 256};
  float *mnist_model_2_weight =
      read_file("./binary/mnist_model_2_weight.bin", 256 * 10);
  float *mnist_model_2_bias = read_file("./binary/mnist_model_2_bias.bin", 10);
  LinearLayer linear2 = {mnist_model_2_weight, mnist_model_2_bias, 256, 10};
  TwoLayerNet model = {linear0, linear2};

  while (!WindowShouldClose()) {
    // Use mouse to draw mnist image
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) ||
        (GetGestureDetected() == GESTURE_DRAG)) {
      Vector2 mouse_position = GetMousePosition();
      int x = (int)mouse_position.x / pixel_size;
      int y = (int)mouse_position.y / pixel_size;

      // Draw mnist image
      BeginTextureMode(mnist_texture);
      DrawRectangle(x, y, 1, 1, WHITE);
      EndTextureMode();
    }

    // Get mnist image data
    Image mnist_image = LoadImageFromTexture(mnist_texture.texture);
    ImageFormat(&mnist_image, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);
    ImageFlipVertical(&mnist_image);
    int digit = predict_digit(mnist_image.data, &model);

    // Draw mnist image to window
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawTexturePro(mnist_texture.texture,
                   (Rectangle){0, 0, mnist_size, -mnist_size},
                   (Rectangle){0, 0, window_width, window_height},
                   (Vector2){0, 0}, 0, WHITE);
    DrawText("Draw an number", 5, 5, 20, GRAY);
    char text[32];
    snprintf(text, 32, "Predicted: %d", digit);
    DrawText(text, 5, window_height - 25, 20, GRAY);
    EndDrawing();
  }

  UnloadRenderTexture(mnist_texture);
  CloseWindow();

  return 0;
}
