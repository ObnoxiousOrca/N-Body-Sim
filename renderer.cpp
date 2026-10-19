
#include "renderer.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define AU 1.495978707e11

struct Renderer {
    SDL_Window* window;
    SDL_Renderer* canvas;

    int width;
    int height;

    double zoom;
    double cameraX;
    double cameraY;

    TTF_Font* font;

    bool launchMode;
    bool positionSelected;
    bool requestReady;

    double launchXAU;
    double launchYAU;

    int activeField;

    char fields[4][64];
    LaunchRequest pendingRequest;

    std::vector<int> radiusOverrides;
};

Renderer* renderer_create(int width, int height) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return nullptr;

    if (TTF_Init() != 0) {
        SDL_Quit();
        return nullptr;
    }

    Renderer* r = new Renderer{
        nullptr, nullptr,
        width, height,
        12.0, 0.0, 0.0,
        nullptr,
        false, false, false,
        0.0, 0.0,
        0,
        {
            "1e12",   // Mass in kg
            "0",      // vx in km/s
            "30",     // vy in km/s
            "4"       // Radius in pixels
        },
        {0.0, 0.0, 0.0, 0.0, 0.0, 4},
        {}
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

    r->font = TTF_OpenFont(
        "assets/fonts/DejaVuSans.ttf", 16
    );

    if (!r->window || !r->canvas || !r->font) {
        if (r->canvas)
            SDL_DestroyRenderer(r->canvas);
        if (r->window)
            SDL_DestroyWindow(r->window);

        delete r;
        TTF_Quit();
        SDL_Quit();
        return nullptr;
    }

    r->cameraX = 0.0;
    r->cameraY = 0.0;
    r->zoom = 12.0;

    return r;
}

static void drawText(Renderer* r, const char* text, int x, int y, SDL_Color color) {
    if (!r->font) return;

    SDL_Surface* surface =
        TTF_RenderUTF8_Blended(r->font, text, color);

    if (!surface) return;

    SDL_Texture* texture =
        SDL_CreateTextureFromSurface(r->canvas, surface);

    if (texture) {
        SDL_Rect dst = {x, y, surface->w, surface->h};
        SDL_RenderCopy(r->canvas, texture, nullptr, &dst);
        SDL_DestroyTexture(texture);
    }

    SDL_FreeSurface(surface);
}

bool renderer_poll_events(Renderer* r) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT)
            return false;

        if (event.type == SDL_KEYDOWN) {
            SDL_Keycode key = event.key.keysym.sym;

            if (key == SDLK_ESCAPE) {
                if (r->launchMode) {
                    r->launchMode = false;
                    r->positionSelected = false;
                    SDL_StopTextInput();
                } else {
                    return false;
                }
                continue;
            }

            if (key == SDLK_l && !r->launchMode) {
                r->launchMode = true;
                r->positionSelected = false;
                r->activeField = 0;

                // Reset defaults whenever launch mode starts.
                snprintf(r->fields[0], 64, "1e12");
                snprintf(r->fields[1], 64, "0");
                snprintf(r->fields[2], 64, "30");
                snprintf(r->fields[3], 64, "4");

                SDL_StartTextInput();
                continue;
            }

            if (r->launchMode) {
                if (key == SDLK_TAB) {
                    r->activeField = (r->activeField + 1) % 4;
                } else if (key == SDLK_BACKSPACE) {
                    char* field = r->fields[r->activeField];
                    size_t len = strlen(field);

                    if (len > 0)
                        field[len - 1] = '\0';
                } else if (key == SDLK_RETURN ||
                           key == SDLK_KP_ENTER) {

                    double mass = strtod(r->fields[0], nullptr);
                    double vxKm = strtod(r->fields[1], nullptr);
                    double vyKm = strtod(r->fields[2], nullptr);
                    double radiusValue =
                        strtod(r->fields[3], nullptr);

                    bool valid =
                        r->positionSelected &&
                        mass > 0.0 && std::isfinite(mass) &&
                        std::isfinite(vxKm) &&
                        std::isfinite(vyKm) &&
                        radiusValue >= 1.0 &&
                        radiusValue <= 100.0 &&
                        std::floor(radiusValue) == radiusValue;

                    if (valid) {
                        r->pendingRequest = {
                            r->launchXAU * AU,
                            r->launchYAU * AU,
                            vxKm * 1000.0,
                            vyKm * 1000.0,
                            mass,
                            static_cast<int>(radiusValue)
                        };

                        r->requestReady = true;
                        r->launchMode = false;
                        SDL_StopTextInput();
                    }
                }
                continue;
            }

            // Normal camera controls outside launch mode.
            switch (key) {
                case SDLK_LEFT:
                    r->cameraX -= 100.0 / r->zoom;
                    break;
                case SDLK_RIGHT:
                    r->cameraX += 100.0 / r->zoom;
                    break;
                case SDLK_UP:
                    r->cameraY += 100.0 / r->zoom;
                    break;
                case SDLK_DOWN:
                    r->cameraY -= 100.0 / r->zoom;
                    break;
            }
        }

        if (r->launchMode && event.type == SDL_TEXTINPUT) {
            char* field = r->fields[r->activeField];

            size_t len = strlen(field);
            size_t remaining = 63 - len;

            strncat(field, event.text.text, remaining);
        }

        if (event.type == SDL_MOUSEBUTTONDOWN &&
            event.button.button == SDL_BUTTON_LEFT) {

            int mx = event.button.x;
            int my = event.button.y;

            if (r->launchMode) {
                // Launch panel occupies this rectangle.
                bool inPanel =
                    mx >= 20 && mx <= 370 &&
                    my >= 20 && my <= 270;

                if (inPanel) {
                    // Click a field to select it.
                    if (my >= 85 && my < 125)
                        r->activeField = 0;
                    else if (my >= 125 && my < 160)
                        r->activeField = 1;
                    else if (my >= 160 && my < 195)
                        r->activeField = 2;
                    else if (my >= 195 && my < 230)
                        r->activeField = 3;
                } else {
                    // Convert the clicked screen position to AU.
                    r->launchXAU =
                        (mx - r->width / 2.0) / r->zoom
                        + r->cameraX;

                    r->launchYAU =
                        (r->height / 2.0 - my) / r->zoom
                        + r->cameraY;

                    r->positionSelected = true;
                }
            }
        }

        if (event.type == SDL_MOUSEWHEEL) {
            if (event.wheel.y > 0)
                r->zoom *= 1.25;
            else if (event.wheel.y < 0)
                r->zoom /= 1.25;

            if (r->zoom < 0.05) r->zoom = 0.05;
            if (r->zoom > 500.0) r->zoom = 500.0;
        }
    }

    return true;
}


void renderer_draw(Renderer* r, const double* x, const double* y, const double* mass, int n) {
    SDL_SetRenderDrawColor(r->canvas, 15, 18, 35, 255);
    SDL_RenderClear(r->canvas);

    const double JUPITER_MASS = 1.8982e27;
    const int JUPITER_RADIUS = 12;
    const int SUN_RADIUS = 15;

    double zoom = (r->zoom > 0.0) ? r->zoom : 12.0;

    for (int i = 0; i < n; i++) {
        double worldX = x[i] / AU;
        double worldY = y[i] / AU;

        int sx = (int)(
            (worldX - r->cameraX) * zoom + r->width / 2.0
        );

        int sy = (int)(
            r->height / 2.0 - (worldY - r->cameraY) * zoom
        );

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

        int radius;

        if (i < static_cast<int>(r->radiusOverrides.size()) && r->radiusOverrides[i] > 0) {
            radius = r->radiusOverrides[i];
        } 
        else if (i == 0) radius = SUN_RADIUS;
        else if (mass[i] > 0.0) {
            radius = static_cast<int>(
                round(JUPITER_RADIUS *
                    cbrt(mass[i] / JUPITER_MASS))
            );
        } else {
            radius = 1;
        }

        // Draw a filled circle.
        for (int dy = -radius; dy <= radius; dy++) {
            int dx = (int)sqrt(
                (double)radius * radius - (double)dy * dy
            );

            SDL_RenderDrawLine(
                r->canvas,
                sx - dx, sy + dy,
                sx + dx, sy + dy
            );
        }
    }

    if (r->launchMode) {
        SDL_Rect panel = {20, 20, 350, 250};

        SDL_SetRenderDrawColor(r->canvas, 28, 30, 52, 245);
        SDL_RenderFillRect(r->canvas, &panel);

        SDL_SetRenderDrawColor(r->canvas, 150, 130, 255, 255);
        SDL_RenderDrawRect(r->canvas, &panel);

        SDL_Color white = {240, 240, 255, 255};
        SDL_Color purple = {190, 175, 255, 255};

        drawText(r, "LAUNCH BODY", 32, 30, purple);

        char positionText[128];

        if (r->positionSelected) {
            snprintf(positionText, sizeof(positionText),
                    "Position: %.3f, %.3f AU",
                    r->launchXAU, r->launchYAU);
        } else {
            snprintf(positionText, sizeof(positionText),
                    "Click outside panel to choose position");
        }

        drawText(r, positionText, 32, 57, white);

        const char* labels[] = {
            "Mass (kg):",
            "Velocity X (km/s):",
            "Velocity Y (km/s):",
            "Radius (px):"
        };

        for (int i = 0; i < 4; i++) {
            int y = 90 + i * 35;

            if (i == r->activeField) {
                SDL_Rect highlight = {28, y - 3, 330, 29};
                SDL_SetRenderDrawColor(r->canvas, 65, 55, 105, 255);
                SDL_RenderFillRect(r->canvas, &highlight);
            }

            drawText(r, labels[i], 32, y, white);
            drawText(r, r->fields[i], 235, y, purple);
        }

        drawText(r, "Tab: next field | Enter: launch",
                32, 232, white);
        drawText(r, "L: cancel/exit | Esc: cancel",
                32, 250, white);
    }

    SDL_RenderPresent(r->canvas);
}

bool renderer_take_launch_request(Renderer* r, LaunchRequest* request) {
    if (!r || !request || !r->requestReady)
        return false;

    *request = r->pendingRequest;
    r->requestReady = false;

    return true;
}

void renderer_set_body_radius(Renderer* r, int bodyIndex, int radius) {
    if (!r || bodyIndex < 0 || radius < 1)
        return;

    if (r->radiusOverrides.size() <=
        static_cast<size_t>(bodyIndex)) {
        r->radiusOverrides.resize(bodyIndex + 1, 0);
    }

    r->radiusOverrides[bodyIndex] = radius;
}

void renderer_delay(unsigned int milliseconds) {
    SDL_Delay(milliseconds);
}

void renderer_destroy(Renderer* r) {
    if (!r)
        return;

    if(r->font) TTF_CloseFont(r->font);

    SDL_DestroyRenderer(r->canvas);
    SDL_DestroyWindow(r->window);
    delete r;

    TTF_Quit();
    SDL_Quit();
}
