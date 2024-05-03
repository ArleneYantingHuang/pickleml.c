#include "raylib.h"
#include "stdint.h"
#include <stdio.h>

int predict_digit(uint8_t *image) {
  // TOOD: Implement neural network

  return 0;
}

int main(int argc, char* argv[]) {
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
  ClearBackground(GetColor(0x000000));
  EndTextureMode();

  while (!WindowShouldClose()) {
    // Use mouse to draw mnist image
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) || (GetGestureDetected() == GESTURE_DRAG)) {
      Vector2 mouse_position = GetMousePosition();
      int x = (int)mouse_position.x / pixel_size;
      int y = (int)mouse_position.y / pixel_size;

      // Draw mnist image
      BeginTextureMode(mnist_texture);
      DrawRectangle(x, y, 1, 1, BLACK);
      EndTextureMode();
    }

    // Get mnist image data
    Image mnist_image = LoadImageFromTexture(mnist_texture.texture);
    ImageFormat(&mnist_image, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);
    int digit = predict_digit(mnist_image.data);

    // Draw mnist image to window
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawTexturePro(mnist_texture.texture, (Rectangle){0, 0, mnist_size, -mnist_size}, (Rectangle){0, 0, window_width, window_height}, (Vector2){0, 0}, 0, WHITE);
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
