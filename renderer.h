
#ifndef RENDERER_H
#define RENDERER_H

struct Renderer;

// Create the SDL2 window and rendering context
Renderer* renderer_create(int width, int height);

// Process keyboard, mouse and window events.
// Returns false when the user requests exit.
bool renderer_poll_events(Renderer* renderer);

// Draw the current body positions.
// x and y are ordinary CPU arrays.
void renderer_draw(
    Renderer* renderer,
    const double* x,
    const double* y,
    int n
);

void renderer_delay(unsigned int milliseconds);

// Release SDL2 resources
void renderer_destroy(Renderer* renderer);

#endif
