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

// Live numbers shown in the on-screen HUD (set every frame from main.cu).
struct HudStats {
    bool gpuMode;    // true = parallel (CUDA), false = sequential (CPU)
    double cpuMs;    // smoothed ms per step in CPU mode, < 0 if not measured yet
    double gpuMs;    // smoothed ms per step in GPU mode, < 0 if not measured yet
    double fps;      // frames per second of the whole loop
    int bodies;      // current body count
};

Renderer* renderer_create(int width, int height);
bool renderer_poll_events(Renderer* renderer);

bool renderer_take_launch_request(
    Renderer* renderer,
    LaunchRequest* request
);

// Returns true once each time the mode button (or the M key) is pressed.
bool renderer_take_mode_toggle(Renderer* renderer);

void renderer_set_hud(Renderer* renderer, const HudStats* stats);

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
