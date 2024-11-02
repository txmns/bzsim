#include <SDL2/SDL.h>

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "common.c"
// #include "bzsim_naive.c"
// #include "bzsim_simd.c"
// #include "bzsim_simd_unrolled.c"
#include "bzsim_simd_unrolled_parallel.c"

#define FPS 60
#define FrameTargetDurationMs (1000/FPS)

typedef struct SdlResources SdlResources;
struct SdlResources
{
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture* buffer;
};

internal SdlResources
sdl_init(S32 width, S32 height)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL ERROR: %s\n", SDL_GetError());
        exit(1);
    }
    SDL_Window *windows = SDL_CreateWindow("BZSim",
                                           SDL_WINDOWPOS_CENTERED,
                                           SDL_WINDOWPOS_CENTERED,
                                           width,
                                           height,
                                           SDL_WINDOW_RESIZABLE);
    if (!windows) {
        fprintf(stderr, "SDL ERROR: %s\n", SDL_GetError());
        exit(1);
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(windows, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "SDL ERROR: %s\n", SDL_GetError());
        exit(1);
    }
    if (SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND) < 0) {
        fprintf(stderr, "SDL ERROR: %s\n", SDL_GetError());
        exit(1);
    }
    SDL_Texture* buffer = SDL_CreateTexture(renderer,
                                            SDL_PIXELFORMAT_RGBA8888,
                                            SDL_TEXTUREACCESS_STREAMING,
                                            width,
                                            height);
    if (!buffer) {
        fprintf(stderr, "SDL ERROR: %s\n", SDL_GetError());
        exit(1);
    }
    SdlResources result = {
        .window = windows,
        .renderer = renderer,
        .buffer = buffer,
    };
    return result;
}

internal void
sdl_free(SdlResources *sdl)
{
    SDL_DestroyTexture(sdl->buffer);
    SDL_DestroyRenderer(sdl->renderer);
    SDL_DestroyWindow(sdl->window);
    SDL_Quit();
}

int
main(int argc, char **argv)
{
    S32 width = 1600; //1280;
    S32 height = 900; //720;

    F32 alpha = 1.4f;
    F32 betta = 1.0f;
    F32 gamma = 1.0f;

    BZSim bz = bzsim_init(width, height, alpha, betta, gamma);

    SdlResources sdl = sdl_init(width, height);
    B32 nextFrame = False;
    B32 pause = False;
    B32 quit = False;
    while (!quit) {
        U64 startMs = SDL_GetTicks64();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT: {
                    quit = True;
                } break;
                case SDL_KEYDOWN: {
                    if (event.key.keysym.sym == SDLK_ESCAPE) {
                        quit = True;
                    }
                    if (event.key.keysym.sym == SDLK_SPACE) {
                        pause = !pause;
                    }
                    if (event.key.keysym.sym == SDLK_RIGHT) {
                        pause = True;
                        nextFrame = True;
                    }
                } break;
            }
        }

        U32 *pixels;
        S32 pitch;
        if (SDL_LockTexture(sdl.buffer, NULL, (void **)&pixels, &pitch) < 0) {
            fprintf(stderr, "SDL ERROR: %s\n", SDL_GetError());
            exit(1);
        }

        if (!pause || nextFrame) {
            bzsim_update(&bz, pixels);
            nextFrame = False;
        }

        SDL_UnlockTexture(sdl.buffer);

        SDL_RenderClear(sdl.renderer);
        SDL_RenderCopy(sdl.renderer, sdl.buffer, NULL, NULL);
        SDL_RenderPresent(sdl.renderer);

        U64 endMs = SDL_GetTicks64();

        U32 durationMs = (U32)(endMs - startMs);
        if (durationMs < FrameTargetDurationMs) {
            SDL_Delay(FrameTargetDurationMs - durationMs);
        } else {
            fprintf(stderr, "Missed a frame by %u ms\n", durationMs);
        }
    }

    sdl_free(&sdl);
    bzsim_free(&bz);

    exit(0);
}
