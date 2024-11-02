#include <stdlib.h>

typedef struct BZSim BZSim;
struct BZSim
{
    S32 width;
    S32 height;

    F32 *A0, *B0, *C0;
    F32 *A1, *B1, *C1;
    F32 alpha, betta, gamma;
};

internal BZSim
bzsim_init(S32 width, S32 height, F32 alpha, F32 betta, F32 gamma)
{
    F32 *A0 = (F32 *)calloc((2 + height) * (2 + width), sizeof(F32));
    F32 *B0 = (F32 *)calloc((2 + height) * (2 + width), sizeof(F32));
    F32 *C0 = (F32 *)calloc((2 + height) * (2 + width), sizeof(F32));
    F32 *A1 = (F32 *)calloc((2 + height) * (2 + width), sizeof(F32));
    F32 *B1 = (F32 *)calloc((2 + height) * (2 + width), sizeof(F32));
    F32 *C1 = (F32 *)calloc((2 + height) * (2 + width), sizeof(F32));

    for (S32 y = 1; y <= height; y++) {
        for (S32 x = 1; x <= width; x++) {
            A0[y*width + x] = rand_uniform();
            B0[y*width + x] = rand_uniform();
            C0[y*width + x] = rand_uniform();
        }
    }

    BZSim result = {
        .width = width,
        .height = height,
        .alpha = alpha,
        .betta = betta,
        .gamma = gamma,
        .A0 = A0,
        .B0 = B0,
        .C0 = C0,
        .A1 = A1,
        .B1 = B1,
        .C1 = C1,
    };
    return result;
}

internal void
bzsim_update(BZSim *bz, U32 *pixels)
{
    // Algorithm based on:
    // Turner, A.; (2009) A simple model of the Belousov-Zhabotinsky reaction from first principles.
    // Bartlett School of Graduate Studies, UCL: London, UK.
    // https://discovery.ucl.ac.uk/id/eprint/17241/

    S32 width = bz->width;
    S32 height = bz->height;

    F32 alpha = bz->alpha;
    F32 betta = bz->betta;
    F32 gamma = bz->gamma;

    F32 *A0 = bz->A0;
    F32 *B0 = bz->B0;
    F32 *C0 = bz->C0;
    F32 *A1 = bz->A1;
    F32 *B1 = bz->B1;
    F32 *C1 = bz->C1;

    for (S32 y = 1; y <= height; y++) {
        for (S32 x = 1; x <= width; x++) {
            F32 a0 = 0.0f;
            F32 b0 = 0.0f;
            F32 c0 = 0.0f;
            for (S32 j = -1; j <= 1; j++) {
                for (S32 i = -1; i <= 1; i++) {
                    a0 += A0[(y+j)*width + (x+i)];
                    b0 += B0[(y+j)*width + (x+i)];
                    c0 += C0[(y+j)*width + (x+i)];
                }
            }
            a0 /= 9.0f;
            b0 /= 9.0f;
            c0 /= 9.0f;

            F32 a1 = a0 + a0*(alpha*b0 - gamma*c0);
            F32 b1 = b0 + b0*(betta*c0 - alpha*a0);
            F32 c1 = c0 + c0*(gamma*a0 - betta*b0);
            a1 = Max(Min(a1, 1.0f), 0.0f);
            b1 = Max(Min(b1, 1.0f), 0.0f);
            c1 = Max(Min(c1, 1.0f), 0.0f);

            U32 r = (U32)(a1*255.0f);
            U32 g = (U32)(b1*255.0f);
            U32 b = (U32)(c1*255.0f);
            U32 a = 255;

            A1[y*width + x] = a1;
            B1[y*width + x] = b1;
            C1[y*width + x] = c1;
            pixels[(y-1)*width + (x-1)] = (U32)((r << 24) | (g << 16) | (b << 8) | a);
        }
    }
    swap(&bz->A0, &bz->A1);
    swap(&bz->B0, &bz->B1);
    swap(&bz->C0, &bz->C1);
}

internal void
bzsim_free(BZSim *bz)
{
    free(bz->A0);
    free(bz->B0);
    free(bz->C0);
    free(bz->A1);
    free(bz->B1);
    free(bz->C1);
}
