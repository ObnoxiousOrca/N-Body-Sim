
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
#define DT 10000.0
#define STEPS 10

// Calculate acceleration using the current positions
__global__ void calcGrav(Bodies *B) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;

    double ax = 0.0, ay = 0.0;

    for (int j = 0; j < N; j++) {
        if (i == j) continue;

        double dx = B->x[j] - B->x[i];
        double dy = B->y[j] - B->y[i];

        double r2 = dx * dx + dy * dy
                 + EPSILON * EPSILON;

        double invR = rsqrtf(r2);
        double invR3 = invR * invR * invR;

        ax += G * B->mass[j] * dx * invR3;
        ay += G * B->mass[j] * dy * invR3;
    }

    B->ax[i] = ax;
    B->ay[i] = ay;
}


// Velocity Verlet: update position and half-step velocity
__global__ void updatePosition(Bodies *B, double dt) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;

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
__global__ void updateVelocity(Bodies *B, double dt) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;

    B->vx[i] += B->ax[i] * (0.5 * dt);
    B->vy[i] += B->ay[i] * (0.5 * dt);
}


int main() {
    Bodies hB, *dB;

    double *dx, *dy, *dvx, *dvy, *dmass, *dax, *day;

    // Allocate host memory
    hB.x    = (double*)malloc(N * sizeof(double));
    hB.y    = (double*)malloc(N * sizeof(double));
    hB.vx   = (double*)malloc(N * sizeof(double));
    hB.vy   = (double*)malloc(N * sizeof(double));
    hB.mass = (double*)malloc(N * sizeof(double));
    hB.ax   = (double*)malloc(N * sizeof(double));
    hB.ay   = (double*)malloc(N * sizeof(double));

    initialize_bodies(&hB);

    // Allocate GPU memory for each array
    cudaMalloc(&dx,    N * sizeof(double));
    cudaMalloc(&dy,    N * sizeof(double));
    cudaMalloc(&dvx,   N * sizeof(double));
    cudaMalloc(&dvy,   N * sizeof(double));
    cudaMalloc(&dmass, N * sizeof(double));
    cudaMalloc(&dax,   N * sizeof(double));
    cudaMalloc(&day,   N * sizeof(double));

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

    // Compute initial acceleration a(t)
    calcGrav<<<1, N>>>(dB);
    cudaDeviceSynchronize();

    // printf("Time(s)\tBody\tX(m)\t\tY(m)\n");
    Renderer* renderer = renderer_create(1000, 800);

    if (!renderer) {
        printf("Failed to initialize SDL2\n");
        return 1;
    }

    
    bool running = true;

    while (running) {
        // 1. Process SDL2 events
        running = renderer_poll_events(renderer);

        if (!running)
            break;

        // 2. Advance physics by one Velocity Verlet step
        updatePosition<<<(N + 255) / 256, 256>>>(dB, DT);

        calcGrav<<<(N + 255) / 256, 256>>>(dB);

        updateVelocity<<<(N + 255) / 256, 256>>>(dB, DT);

        // Ensure the physics step is complete before reading positions
        cudaDeviceSynchronize();

        // 3. Copy positions from GPU to CPU
        cudaMemcpy(
            hB.x, dx, N * sizeof(double),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            hB.y, dy, N * sizeof(double),
            cudaMemcpyDeviceToHost
        );

        // 4. Render the positions
        renderer_draw(renderer, hB.x, hB.y, N);

        // Optional: cap the frame rate to reduce CPU usage
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
