#ifndef RENDERER_H
#define RENDERER_H

struct Renderer;

struct LaunchRequest {
    double x;       // Position in metres
    double y;       // Position in metres
    double vx;      // Velocity in m/s
    double vy;      // Velocity in m/s
    double mass;    // Mass in kg
    int radius;     // Rendered radius in pixels
};

Renderer* renderer_create(int width, int height);
bool renderer_poll_events(Renderer* renderer);

bool renderer_take_launch_request(
    Renderer* renderer,
    LaunchRequest* request
);

void renderer_set_body_radius(
    Renderer* renderer,
    int bodyIndex,
    int radius
);

void renderer_draw(
    Renderer* renderer,
    const double* x,
    const double* y,
    const double* mass,
    int n
);

void renderer_delay(unsigned int milliseconds);
void renderer_destroy(Renderer* renderer);

#endif
