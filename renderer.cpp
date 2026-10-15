
#include "renderer.h"
#include <SDL2/SDL.h>
#include <cmath>

#define AU 1.495978707e11

struct Renderer {
    SDL_Window* window;
    SDL_Renderer* canvas;

    int width;
    int height;

    double zoom;
    double cameraX;
    double cameraY;
};

Renderer* renderer_create(int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return nullptr;

    Renderer* r = new Renderer{
        nullptr, nullptr,
        width, height,
        12.0, 0.0, 0.0
    };

    r->window = SDL_CreateWindow(
        "CUDA N-Body Simulation",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_SHOWN
    );

    if (r->window) {
        r->canvas = SDL_CreateRenderer(
            r->window, -1, SDL_RENDERER_ACCELERATED
        );
    }

    if (!r->window || !r->canvas) {
        if (r->canvas)
            SDL_DestroyRenderer(r->canvas);
        if (r->window)
            SDL_DestroyWindow(r->window);

        delete r;
        SDL_Quit();
        return nullptr;
    }

    r->cameraX = 0.0;
    r->cameraY = 0.0;
    r->zoom = 12.0;

    return r;
}


bool renderer_poll_events(Renderer* r) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
            return false;

        if (event.type == SDL_KEYDOWN) {
            switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    return false;

                case SDLK_LEFT:
                    r->cameraX -= 100.0f / r->zoom;
                    break;

                case SDLK_RIGHT:
                    r->cameraX += 100.0f / r->zoom;
                    break;

                case SDLK_UP:
                    r->cameraY += 100.0f / r->zoom;
                    break;

                case SDLK_DOWN:
                    r->cameraY -= 100.0f / r->zoom;
                    break;
            }
        }
  
        if (event.type == SDL_MOUSEWHEEL) {
            if (event.wheel.y > 0)
                r->zoom *= 1.25;
            else if (event.wheel.y < 0)
                r->zoom /= 1.25;

            // Minimum and maximum zoom, in pixels per AU.
            if (r->zoom < 0.05)
                r->zoom = 0.05;

            if (r->zoom > 500.0)
                r->zoom = 500.0;
        }

    }

    return true;
}


void renderer_draw(
    Renderer* r,
    const double* x,
    const double* y,
    int n
) {
    SDL_SetRenderDrawColor(r->canvas, 15, 18, 35, 255);
    SDL_RenderClear(r->canvas);

    // Use the existing camera zoom, but prevent a zero or negative scale.
    double zoom = (r->zoom > 0.0) ? r->zoom : 12.0;

    for (int i = 0; i < n; i++) {
        // Convert metres to astronomical units.
        double worldX = x[i] / AU;
        double worldY = y[i] / AU;

        // Convert AU coordinates to screen pixels.
        int sx = (int)(
            (worldX - r->cameraX) * zoom + r->width / 2.0
        );

        int sy = (int)(
            r->height / 2.0 - (worldY - r->cameraY) * zoom
        );

        // Skip bodies outside the window.
        if (sx < 0 || sx >= r->width ||
            sy < 0 || sy >= r->height) {
            continue;
        }

        // Colour bodies cyclically.
        switch (i % 5) {
            case 0:
                SDL_SetRenderDrawColor(r->canvas, 255, 210, 70, 255);
                break;
            case 1:
                SDL_SetRenderDrawColor(r->canvas, 80, 180, 255, 255);
                break;
            case 2:
                SDL_SetRenderDrawColor(r->canvas, 255, 100, 120, 255);
                break;
            case 3:
                SDL_SetRenderDrawColor(r->canvas, 130, 255, 170, 255);
                break;
            default:
                SDL_SetRenderDrawColor(r->canvas, 200, 140, 255, 255);
                break;
        }

        // Make the Sun larger than the other bodies.
        int radius = (i == 0) ? 7 : 3;

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx * dx + dy * dy <= radius * radius) {
                    SDL_RenderDrawPoint(r->canvas, sx + dx, sy + dy);
                }
            }
        }
    }

    SDL_RenderPresent(r->canvas);
}

void renderer_delay(unsigned int milliseconds) {
    SDL_Delay(milliseconds);
}

void renderer_destroy(Renderer* r) {
    if (!r)
        return;

    SDL_DestroyRenderer(r->canvas);
    SDL_DestroyWindow(r->window);
    delete r;

    SDL_Quit();
}
