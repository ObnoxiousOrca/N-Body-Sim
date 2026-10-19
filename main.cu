
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "renderer.h"
#include "bodies.h"
#include "init.h"
#include <stdbool.h>

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

        double invR = rsqrtf(r2);
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

    
    bool running = true;
    bool launched = false;

    while (running) {
        running = renderer_poll_events(renderer);
        if (!running) break;

        LaunchRequest request;

        if (renderer_take_launch_request(renderer, &request)) {
            int oldCount = nBodies;

            if (launchBody(
                    &hB, &nBodies,
                    request.x, request.y,
                    request.vx, request.vy,
                    request.mass)) {

                if (uploadBodyToGPU(&hB, oldCount,dx, dy, dvx, dvy, dmass)) {

                    renderer_set_body_radius(renderer, oldCount, request.radius);

                    // Recalculate acceleration with the new body included.
                    int blocks =(nBodies + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;

                    calcGrav<<<blocks, THREADS_PER_BLOCK>>>(dB, nBodies);

                    cudaError_t err = cudaDeviceSynchronize();

                    if (err != cudaSuccess) {
                        fprintf(stderr, "Launch failed: %s\n",
                                cudaGetErrorString(err));
                        running = false;
                        break;
                    }

                    printf("Launched body %d\n", oldCount);
                } else {
                    nBodies = oldCount;
                    fprintf(stderr, "Failed to upload launched body.\n");
                }
            } else {
                fprintf(stderr, "Maximum body capacity reached.\n");
            }
        }

        if (!launched) {
            int oldCount = nBodies;

            bool success = launchBody(&hB, &nBodies,1.0 * AU, 0.0, 0.0, 30000.0, 1.0e12);

            if (success) {
                int i = oldCount;

                if (uploadBodyToGPU(
                        &hB, i, dx, dy, dvx, dvy, dmass)) {

                    int blocks = (nBodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

                    calcGrav<<<blocks, THREADS_PER_BLOCK>>>(dB, nBodies);
                    cudaDeviceSynchronize();

                    launched = true;
                    printf("Launched body %d\n", i);
                } else {
                    nBodies = oldCount;
                }
            }
        }

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

        size_t bytes = (size_t)nBodies * sizeof(double);

        cudaMemcpy(hB.x, dx, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hB.y, dy, bytes, cudaMemcpyDeviceToHost);

        renderer_draw(renderer, hB.x, hB.y, hB.mass, nBodies);
        renderer_delay(1);
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
