
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "renderer.h"
#include "bodies.h"
#include "init.h"
#include <stdbool.h>
#include <chrono>

#define EPSILON 0.0
#define DT 100000.0
#define STEPS 10
#define MAX_BODIES (N+100)
#define THREADS_PER_BLOCK 256

// Calculate acceleration using the current positions
__global__ void calcGrav(Bodies *B, int nBodies) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nBodies) return;

    double ax = 0.0, ay = 0.0;

    for (int j = 0; j < nBodies; j++) {
        if (i == j) continue;

        double dx = B->x[j] - B->x[i];
        double dy = B->y[j] - B->y[i];

        double r2 = dx * dx + dy * dy
                 + EPSILON * EPSILON;
        if(r2 == 0.0) continue;

        double invR = rsqrt(r2);
        double invR3 = invR * invR * invR;

        ax += G * B->mass[j] * dx * invR3;
        ay += G * B->mass[j] * dy * invR3;
    }

    B->ax[i] = ax;
    B->ay[i] = ay;
}


// Velocity Verlet: update position and half-step velocity
__global__ void updatePosition(Bodies *B, double dt, int nBodies) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nBodies) return;

    double half_dt = 0.5 * dt;

    B->x[i] += B->vx[i] * dt
             + 0.5 * B->ax[i] * dt * dt;

    B->y[i] += B->vy[i] * dt
             + 0.5 * B->ay[i] * dt * dt;

    // Store velocity at the half step
    B->vx[i] += B->ax[i] * half_dt;
    B->vy[i] += B->ay[i] * half_dt;
}


// Complete the velocity update using the new acceleration
__global__ void updateVelocity(Bodies *B, double dt, int nBodies) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nBodies) return;

    B->vx[i] += B->ax[i] * (0.5 * dt);
    B->vy[i] += B->ay[i] * (0.5 * dt);
}

// ---------------------------------------------------------------------
// Sequential (CPU) versions of the three kernels above.
// Same maths, same order of operations -> results match the GPU path.
// ---------------------------------------------------------------------
static void cpuCalcGrav(Bodies *B, int nBodies) {
    const double *__restrict__ px = B->x;
    const double *__restrict__ py = B->y;
    const double *__restrict__ pm = B->mass;

    for (int i = 0; i < nBodies; i++) {
        double ax = 0.0, ay = 0.0;
        const double xi = px[i], yi = py[i];

        for (int j = 0; j < nBodies; j++) {
            if (i == j) continue;

            double dx = px[j] - xi;
            double dy = py[j] - yi;

            double r2 = dx * dx + dy * dy
                     + EPSILON * EPSILON;
            if (r2 == 0.0) continue;

            double invR = 1.0 / sqrt(r2);
            double invR3 = invR * invR * invR;

            ax += G * pm[j] * dx * invR3;
            ay += G * pm[j] * dy * invR3;
        }

        B->ax[i] = ax;
        B->ay[i] = ay;
    }
}

static void cpuUpdatePosition(Bodies *B, double dt, int nBodies) {
    double half_dt = 0.5 * dt;

    for (int i = 0; i < nBodies; i++) {
        B->x[i] += B->vx[i] * dt
                 + 0.5 * B->ax[i] * dt * dt;

        B->y[i] += B->vy[i] * dt
                 + 0.5 * B->ay[i] * dt * dt;

        B->vx[i] += B->ax[i] * half_dt;
        B->vy[i] += B->ay[i] * half_dt;
    }
}

static void cpuUpdateVelocity(Bodies *B, double dt, int nBodies) {
    for (int i = 0; i < nBodies; i++) {
        B->vx[i] += B->ax[i] * (0.5 * dt);
        B->vy[i] += B->ay[i] * (0.5 * dt);
    }
}

bool launchBody(Bodies *B, int *nBodies, 
    double x, double y, 
    double vx, double vy, 
    double mass
) {
    if (*nBodies >= MAX_BODIES)
        return false;

    int i = (*nBodies)++;

    B->x[i] = x;
    B->y[i] = y;
    B->vx[i] = vx;
    B->vy[i] = vy;
    B->mass[i] = mass;
    B->ax[i] = 0.0;
    B->ay[i] = 0.0;

    return true;
}

bool uploadBodyToGPU(
    const Bodies *B,
    int i,
    double *dx, double *dy,
    double *dvx, double *dvy,
    double *dmass
) {
    cudaError_t err;

    #define COPY_FIELD(device, host) \
        do { \
            err = cudaMemcpy((device) + i, (host) + i, \
                             sizeof(double), \
                             cudaMemcpyHostToDevice); \
            if (err != cudaSuccess) { \
                fprintf(stderr, "CUDA copy failed: %s\n", \
                        cudaGetErrorString(err)); \
                return false; \
            } \
        } while (0)

    COPY_FIELD(dx, B->x);
    COPY_FIELD(dy, B->y);
    COPY_FIELD(dvx, B->vx);
    COPY_FIELD(dvy, B->vy);
    COPY_FIELD(dmass, B->mass);

    #undef COPY_FIELD
    return true;
}

// Copy the complete simulation state (x, y, vx, vy, mass, ax, ay)
// between host and device. Used when switching CPU <-> GPU mode.
// Array order must match hPtr / dPtr in main().
static bool syncState(double *const host[7], double *const dev[7],
                      int nBodies, cudaMemcpyKind kind) {
    size_t bytes = (size_t)nBodies * sizeof(double);

    for (int k = 0; k < 7; k++) {
        cudaError_t err = (kind == cudaMemcpyHostToDevice)
            ? cudaMemcpy(dev[k], host[k], bytes, kind)
            : cudaMemcpy(host[k], dev[k], bytes, kind);

        if (err != cudaSuccess) {
            fprintf(stderr, "State sync failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

// A new body was just appended on the host at index i.
// Make the ACTIVE backend aware of it and refresh accelerations.
static bool registerNewBody(bool useGpu, Bodies *hB, Bodies *dB,
                            double *const dev[7], int i, int nBodies) {
    if (useGpu) {
        if (!uploadBodyToGPU(hB, i, dev[0], dev[1], dev[2], dev[3], dev[4]))
            return false;

        int blocks = (nBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        calcGrav<<<blocks, THREADS_PER_BLOCK>>>(dB, nBodies);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    } else {
        // Host copy is the source of truth in CPU mode.
        cpuCalcGrav(hB, nBodies);
    }
    return true;
}

int main() {
    Bodies hB, *dB;

    double *dx, *dy, *dvx, *dvy, *dmass, *dax, *day;
    int nBodies = N;

    // Allocate host memory
    hB.x    = (double*)malloc(MAX_BODIES * sizeof(double));
    hB.y    = (double*)malloc(MAX_BODIES * sizeof(double));
    hB.vx   = (double*)malloc(MAX_BODIES * sizeof(double));
    hB.vy   = (double*)malloc(MAX_BODIES * sizeof(double));
    hB.mass = (double*)malloc(MAX_BODIES * sizeof(double));
    hB.ax   = (double*)malloc(MAX_BODIES * sizeof(double));
    hB.ay   = (double*)malloc(MAX_BODIES * sizeof(double));

    initialize_bodies(&hB);

    // Allocate GPU memory for each array
    cudaMalloc(&dx,    MAX_BODIES * sizeof(double));
    cudaMalloc(&dy,    MAX_BODIES * sizeof(double));
    cudaMalloc(&dvx,   MAX_BODIES * sizeof(double));
    cudaMalloc(&dvy,   MAX_BODIES * sizeof(double));
    cudaMalloc(&dmass, MAX_BODIES * sizeof(double));
    cudaMalloc(&dax,   MAX_BODIES * sizeof(double));
    cudaMalloc(&day,   MAX_BODIES * sizeof(double));

    // Copy initial state to GPU
    cudaMemcpy(dx, hB.x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hB.y, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dvx, hB.vx, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dvy, hB.vy, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dmass, hB.mass, N * sizeof(double), cudaMemcpyHostToDevice);

    // Device-side structure contains device pointers
    Bodies dBodies = {dx, dy, dvx, dvy, dmass, dax, day};

    cudaMalloc(&dB, sizeof(Bodies));
    cudaMemcpy(dB, &dBodies, sizeof(Bodies), cudaMemcpyHostToDevice);

    int blocks = (nBodies + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // Compute initial acceleration a(t)
    calcGrav<<<blocks, THREADS_PER_BLOCK>>>(dB, nBodies);
    cudaDeviceSynchronize();

    // printf("Time(s)\tBody\tX(m)\t\tY(m)\n");
    Renderer* renderer = renderer_create(1000, 800);

    if (!renderer) {
        printf("Failed to initialize SDL2\n");
        return 1;
    }

    
    // Pointer tables so whole-state copies can be done in a loop.
    // Order: x, y, vx, vy, mass, ax, ay
    double *hPtr[7] = {hB.x, hB.y, hB.vx, hB.vy, hB.mass, hB.ax, hB.ay};
    double *dPtr[7] = {dx, dy, dvx, dvy, dmass, dax, day};

    using Clock = std::chrono::steady_clock;

    bool useGpu = true;       // start in parallel mode
    double cpuMs = -1.0;      // smoothed ms/step, sequential (-1 = no data)
    double gpuMs = -1.0;      // smoothed ms/step, parallel   (-1 = no data)
    double frameMs = 16.0;    // smoothed ms/frame, for the FPS readout
    int skipSamples = 2;      // ignore warm-up frames after a mode change

    bool running = true;
    bool launched = false;

    while (running) {
        auto frameStart = Clock::now();

        running = renderer_poll_events(renderer);
        if (!running) break;

        // ---- Sequential <-> parallel toggle (button or M key) ----
        if (renderer_take_mode_toggle(renderer)) {
            bool ok = useGpu
                ? syncState(hPtr, dPtr, nBodies, cudaMemcpyDeviceToHost)
                : syncState(hPtr, dPtr, nBodies, cudaMemcpyHostToDevice);

            if (ok) {
                useGpu = !useGpu;
                skipSamples = 1;
                printf("Switched to %s mode\n",
                       useGpu ? "GPU (parallel)" : "CPU (sequential)");
            }
        }

        LaunchRequest request;

        if (renderer_take_launch_request(renderer, &request)) {
            int oldCount = nBodies;

            if (launchBody(
                    &hB, &nBodies,
                    request.x, request.y,
                    request.vx, request.vy,
                    request.mass)) {

                if (registerNewBody(useGpu, &hB, dB, dPtr, oldCount, nBodies)) {
                    renderer_set_body_radius(renderer, oldCount, request.radius);
                    printf("Launched body %d\n", oldCount);
                } else {
                    nBodies = oldCount;
                    fprintf(stderr, "Failed to add launched body.\n");
                }
            } else {
                fprintf(stderr, "Maximum body capacity reached.\n");
            }
        }

        if (!launched) {
            int oldCount = nBodies;

            bool success = launchBody(&hB, &nBodies, 1.0 * AU, 0.0, 0.0, 30000.0, 1.0e12);

            if (success) {
                if (registerNewBody(useGpu, &hB, dB, dPtr, oldCount, nBodies)) {
                    launched = true;
                    printf("Launched body %d\n", oldCount);
                } else {
                    nBodies = oldCount;
                }
            }
        }

        // ---- One simulation step, timed ----
        auto stepStart = Clock::now();

        if (useGpu) {
            int blocks = (nBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            updatePosition<<<blocks, THREADS_PER_BLOCK>>>(dB, DT, nBodies);
            calcGrav<<<blocks, THREADS_PER_BLOCK>>>(dB, nBodies);
            updateVelocity<<<blocks, THREADS_PER_BLOCK>>>(dB, DT, nBodies);

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n",
                        cudaGetErrorString(err));
                break;
            }

            // Positions must come back to the host so they can be drawn.
            size_t bytes = (size_t)nBodies * sizeof(double);
            cudaMemcpy(hB.x, dx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(hB.y, dy, bytes, cudaMemcpyDeviceToHost);
        } else {
            cpuUpdatePosition(&hB, DT, nBodies);
            cpuCalcGrav(&hB, nBodies);
            cpuUpdateVelocity(&hB, DT, nBodies);
        }

        double stepMs = std::chrono::duration<double, std::milli>(
                            Clock::now() - stepStart).count();

        if (skipSamples > 0) {
            skipSamples--;
        } else {
            double &avg = useGpu ? gpuMs : cpuMs;
            avg = (avg < 0.0) ? stepMs : 0.8 * avg + 0.2 * stepMs;
        }

        HudStats hud = {useGpu, cpuMs, gpuMs, 1000.0 / frameMs, nBodies};
        renderer_set_hud(renderer, &hud);

        renderer_draw(renderer, hB.x, hB.y, hB.mass, nBodies);
        renderer_delay(1);

        double thisFrameMs = std::chrono::duration<double, std::milli>(
                                 Clock::now() - frameStart).count();
        frameMs = 0.9 * frameMs + 0.1 * thisFrameMs;
    }

    // 5. Shut down the renderer
    renderer_destroy(renderer);

    // Then free your CUDA and host allocations as before.


    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dvx);
    cudaFree(dvy);
    cudaFree(dmass);
    cudaFree(dax);
    cudaFree(day);
    cudaFree(dB);

    free(hB.x);
    free(hB.y);
    free(hB.vx);
    free(hB.vy);
    free(hB.mass);
    free(hB.ax);
    free(hB.ay);

    return 0;
}
