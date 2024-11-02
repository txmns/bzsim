#include <immintrin.h>
#include <stdlib.h>

typedef struct BZSim BZSim;
struct BZSim
{
    S32 width;
    S32 height;

    __m512 cAvgDiv;
    __m512 cAbgz;
    __m512 cGabz;
    __m512 cMax;
    __m512 cMin;
    __m512 cRgbaMult;
    __m512 cRgbaAdd;
    __m512i cRgbaLeftShiftCount;
    __m512i cRgbaPermuteIdx;
    __m512i cRgbaPermuteIdx_4x;
    __m512i cRgbaPermuteIdx_8x;
    __m512i cRgbaPermuteIdx_12x;

    F32 *ABCZ0;
    F32 *ABCZ1;
};

internal BZSim
bzsim_init(S32 width, S32 height, F32 alpha, F32 betta, F32 gamma)
{
    Assert(width % 4 == 0);

    F32 *ABCZ1 = (F32 *)calloc((2 + height) * (2 + width) * 4, sizeof(F32));
    F32 *ABCZ0 = (F32 *)calloc((2 + height) * (2 + width) * 4, sizeof(F32));

    for (S32 y = 1; y <= height; y++) {
        for (S32 x = 1; x <= width; x+=4) {
            __m512 randVec = _mm512_set_ps(0.0f, rand_uniform(), rand_uniform(), rand_uniform(),
                                           0.0f, rand_uniform(), rand_uniform(), rand_uniform(),
                                           0.0f, rand_uniform(), rand_uniform(), rand_uniform(),
                                           0.0f, rand_uniform(), rand_uniform(), rand_uniform());
            _mm512_storeu_ps(ABCZ0 + (y*width + x)*4, randVec);
        }
    }

    __m512 cAvgDiv = _mm512_set_ps(1.0f, 9.0f, 9.0f, 9.0f,
                                   1.0f, 9.0f, 9.0f, 9.0f,
                                   1.0f, 9.0f, 9.0f, 9.0f,
                                   1.0f, 9.0f, 9.0f, 9.0f);
    __m512 cAbgz = _mm512_set_ps(0.0f, gamma, betta, alpha,
                                 0.0f, gamma, betta, alpha,
                                 0.0f, gamma, betta, alpha,
                                 0.0f, gamma, betta, alpha);
    __m512 cGabz = _mm512_set_ps(0.0f, betta, alpha, gamma,
                                 0.0f, betta, alpha, gamma,
                                 0.0f, betta, alpha, gamma,
                                 0.0f, betta, alpha, gamma);
    __m512 cMax = _mm512_set_ps(0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f);
    __m512 cMin = _mm512_set_ps(0.0f, 1.0f, 1.0f, 1.0f,
                                0.0f, 1.0f, 1.0f, 1.0f,
                                0.0f, 1.0f, 1.0f, 1.0f,
                                0.0f, 1.0f, 1.0f, 1.0f);
    __m512 cRgbaMult = _mm512_set_ps(0.0f, 255.0f, 255.0f, 255.0f,
                                     0.0f, 255.0f, 255.0f, 255.0f,
                                     0.0f, 255.0f, 255.0f, 255.0f,
                                     0.0f, 255.0f, 255.0f, 255.0f);
    __m512 cRgbaAdd = _mm512_set_ps(255.0f, 0.0f, 0.0f, 0.0f,
                                    255.0f, 0.0f, 0.0f, 0.0f,
                                    255.0f, 0.0f, 0.0f, 0.0f,
                                    255.0f, 0.0f, 0.0f, 0.0f);
    __m512i cRgbaLeftShiftCount = _mm512_set_epi32(0, 8, 16, 24,
                                                   0, 8, 16, 24,
                                                   0, 8, 16, 24,
                                                   0, 8, 16, 24);
    __m512i cRgbaPermuteIdx = _mm512_set_epi32(0, 0, 0, 0,
                                               0, 0, 0, 0,
                                               0, 0, 0, 0,
                                               15, 11, 7, 3);
    __m512i cRgbaPermuteIdx_4x = _mm512_set_epi32(0, 0, 0, 0,
                                                  0, 0, 0, 0,
                                                  15, 11, 7, 3,
                                                  0, 0, 0, 0);
    __m512i cRgbaPermuteIdx_8x = _mm512_set_epi32(0, 0, 0, 0,
                                                  15, 11, 7, 3,
                                                  0, 0, 0, 0,
                                                  0, 0, 0, 0);
    __m512i cRgbaPermuteIdx_12x = _mm512_set_epi32(15, 11, 7, 3,
                                                   0, 0, 0, 0,
                                                   0, 0, 0, 0,
                                                   0, 0, 0, 0);

    BZSim result = {
        .width = width,
        .height = height,
        .cAvgDiv = cAvgDiv,
        .cAbgz = cAbgz,
        .cGabz = cGabz,
        .cMax = cMax,
        .cMin = cMin,
        .cRgbaMult = cRgbaMult,
        .cRgbaAdd = cRgbaAdd,
        .cRgbaLeftShiftCount = cRgbaLeftShiftCount,
        .cRgbaPermuteIdx = cRgbaPermuteIdx,
        .cRgbaPermuteIdx_4x = cRgbaPermuteIdx_4x,
        .cRgbaPermuteIdx_8x = cRgbaPermuteIdx_8x,
        .cRgbaPermuteIdx_12x = cRgbaPermuteIdx_12x,
        .ABCZ0 = ABCZ0,
        .ABCZ1 = ABCZ1,
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

    __m512 cAvgDiv = bz->cAvgDiv;
    __m512 cAbgz = bz->cAbgz;
    __m512 cGabz = bz->cGabz;
    __m512 cMax = bz->cMax;
    __m512 cMin = bz->cMin;
    __m512 cRgbaMult = bz->cRgbaMult;
    __m512 cRgbaAdd = bz->cRgbaAdd;
    __m512i cRgbaLeftShiftCount = bz->cRgbaLeftShiftCount;
    __m512i cRgbaPermuteIdx = bz->cRgbaPermuteIdx;
    __m512i cRgbaPermuteIdx_4x = bz->cRgbaPermuteIdx_4x;
    __m512i cRgbaPermuteIdx_8x = bz->cRgbaPermuteIdx_8x;
    __m512i cRgbaPermuteIdx_12x = bz->cRgbaPermuteIdx_12x;

    F32 *ABCZ0 = bz->ABCZ0;
    F32 *ABCZ1 = bz->ABCZ1;

    for (S32 y = 1; y <= height; y++) {
        for (S32 x = 1; x <= width; x+=16) {
            // abcz0_i = [a0_0i, b0_0i, c0_0i, 0,
            //            a0_1i, b0_1i, c0_1i, 0,
            //            a0_2i, b0_2i, c0_2i, 0,
            //            a0_3i, b0_3i, c0_3i, 0]
            __m512 abcz0_0 = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x-1))*4));
            __m512 abcz0_1 = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+0))*4));
            __m512 abcz0_2 = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+1))*4));
            __m512 abcz0_3 = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x-1))*4));
            __m512 abcz0_4 = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+0))*4));
            __m512 abcz0_5 = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+1))*4));
            __m512 abcz0_6 = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x-1))*4));
            __m512 abcz0_7 = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+0))*4));
            __m512 abcz0_8 = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+1))*4));
            // unrolled x+4
            __m512 abcz0_0_4x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x-1+4))*4));
            __m512 abcz0_1_4x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+0+4))*4));
            __m512 abcz0_2_4x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+1+4))*4));
            __m512 abcz0_3_4x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x-1+4))*4));
            __m512 abcz0_4_4x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+0+4))*4));
            __m512 abcz0_5_4x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+1+4))*4));
            __m512 abcz0_6_4x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x-1+4))*4));
            __m512 abcz0_7_4x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+0+4))*4));
            __m512 abcz0_8_4x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+1+4))*4));
            // unrolled x+8
            __m512 abcz0_0_8x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x-1+8))*4));
            __m512 abcz0_1_8x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+0+8))*4));
            __m512 abcz0_2_8x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+1+8))*4));
            __m512 abcz0_3_8x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x-1+8))*4));
            __m512 abcz0_4_8x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+0+8))*4));
            __m512 abcz0_5_8x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+1+8))*4));
            __m512 abcz0_6_8x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x-1+8))*4));
            __m512 abcz0_7_8x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+0+8))*4));
            __m512 abcz0_8_8x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+1+8))*4));
            // unrolled x+12
            __m512 abcz0_0_12x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x-1+12))*4));
            __m512 abcz0_1_12x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+0+12))*4));
            __m512 abcz0_2_12x = _mm512_loadu_ps(ABCZ0 + (((y-1)*width + (x+1+12))*4));
            __m512 abcz0_3_12x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x-1+12))*4));
            __m512 abcz0_4_12x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+0+12))*4));
            __m512 abcz0_5_12x = _mm512_loadu_ps(ABCZ0 + (((y+0)*width + (x+1+12))*4));
            __m512 abcz0_6_12x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x-1+12))*4));
            __m512 abcz0_7_12x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+0+12))*4));
            __m512 abcz0_8_12x = _mm512_loadu_ps(ABCZ0 + (((y+1)*width + (x+1+12))*4));

            // abczSum_7 = [a0_00 + ... + a0_08,
            //              b0_00 + ... + b0_08,
            //              c0_00 + ... + c0_08,
            //              0,
            //              ... ]
            __m512 abczSum_0 = _mm512_add_ps(abcz0_0, abcz0_1);
            __m512 abczSum_1 = _mm512_add_ps(abcz0_2, abcz0_3);
            __m512 abczSum_2 = _mm512_add_ps(abcz0_4, abcz0_5);
            __m512 abczSum_3 = _mm512_add_ps(abcz0_6, abcz0_7);
            __m512 abczSum_4 = _mm512_add_ps(abczSum_0, abczSum_1);
            __m512 abczSum_5 = _mm512_add_ps(abczSum_2, abczSum_3);
            __m512 abczSum_6 = _mm512_add_ps(abczSum_4, abczSum_5);
            __m512 abczSum_7 = _mm512_add_ps(abczSum_6, abcz0_8);
            // unrolled x+4
            __m512 abczSum_0_4x = _mm512_add_ps(abcz0_0_4x, abcz0_1_4x);
            __m512 abczSum_1_4x = _mm512_add_ps(abcz0_2_4x, abcz0_3_4x);
            __m512 abczSum_2_4x = _mm512_add_ps(abcz0_4_4x, abcz0_5_4x);
            __m512 abczSum_3_4x = _mm512_add_ps(abcz0_6_4x, abcz0_7_4x);
            __m512 abczSum_4_4x = _mm512_add_ps(abczSum_0_4x, abczSum_1_4x);
            __m512 abczSum_5_4x = _mm512_add_ps(abczSum_2_4x, abczSum_3_4x);
            __m512 abczSum_6_4x = _mm512_add_ps(abczSum_4_4x, abczSum_5_4x);
            __m512 abczSum_7_4x = _mm512_add_ps(abczSum_6_4x, abcz0_8_4x);
            // unrolled x+8
            __m512 abczSum_0_8x = _mm512_add_ps(abcz0_0_8x, abcz0_1_8x);
            __m512 abczSum_1_8x = _mm512_add_ps(abcz0_2_8x, abcz0_3_8x);
            __m512 abczSum_2_8x = _mm512_add_ps(abcz0_4_8x, abcz0_5_8x);
            __m512 abczSum_3_8x = _mm512_add_ps(abcz0_6_8x, abcz0_7_8x);
            __m512 abczSum_4_8x = _mm512_add_ps(abczSum_0_8x, abczSum_1_8x);
            __m512 abczSum_5_8x = _mm512_add_ps(abczSum_2_8x, abczSum_3_8x);
            __m512 abczSum_6_8x = _mm512_add_ps(abczSum_4_8x, abczSum_5_8x);
            __m512 abczSum_7_8x = _mm512_add_ps(abczSum_6_8x, abcz0_8_8x);
            // unrolled x+12
            __m512 abczSum_0_12x = _mm512_add_ps(abcz0_0_12x, abcz0_1_12x);
            __m512 abczSum_1_12x = _mm512_add_ps(abcz0_2_12x, abcz0_3_12x);
            __m512 abczSum_2_12x = _mm512_add_ps(abcz0_4_12x, abcz0_5_12x);
            __m512 abczSum_3_12x = _mm512_add_ps(abcz0_6_12x, abcz0_7_12x);
            __m512 abczSum_4_12x = _mm512_add_ps(abczSum_0_12x, abczSum_1_12x);
            __m512 abczSum_5_12x = _mm512_add_ps(abczSum_2_12x, abczSum_3_12x);
            __m512 abczSum_6_12x = _mm512_add_ps(abczSum_4_12x, abczSum_5_12x);
            __m512 abczSum_7_12x = _mm512_add_ps(abczSum_6_12x, abcz0_8_12x);

            // abcz0 = [a0, b0, c0, 0, ... ]
            __m512 abcz0 = _mm512_div_ps(abczSum_7, cAvgDiv);
            // cabz0 = [c0, a0, b0, 0, ... ]
            __m512 cabz0 = _mm512_permute_ps(abcz0, _MM_SHUFFLE(3, 1, 0, 2));
            // bcaz0 = [b0, c0, a0, 0, ... ]
            __m512 bcaz0 = _mm512_permute_ps(abcz0, _MM_SHUFFLE(3, 0, 2, 1));
            // unrolled x+4
            __m512 abcz0_4x = _mm512_div_ps(abczSum_7_4x, cAvgDiv);
            __m512 cabz0_4x = _mm512_permute_ps(abcz0_4x, _MM_SHUFFLE(3, 1, 0, 2));
            __m512 bcaz0_4x = _mm512_permute_ps(abcz0_4x, _MM_SHUFFLE(3, 0, 2, 1));
            // unrolled x+8
            __m512 abcz0_8x = _mm512_div_ps(abczSum_7_8x, cAvgDiv);
            __m512 cabz0_8x = _mm512_permute_ps(abcz0_8x, _MM_SHUFFLE(3, 1, 0, 2));
            __m512 bcaz0_8x = _mm512_permute_ps(abcz0_8x, _MM_SHUFFLE(3, 0, 2, 1));
            // unrolled x+12
            __m512 abcz0_12x = _mm512_div_ps(abczSum_7_12x, cAvgDiv);
            __m512 cabz0_12x = _mm512_permute_ps(abcz0_12x, _MM_SHUFFLE(3, 1, 0, 2));
            __m512 bcaz0_12x = _mm512_permute_ps(abcz0_12x, _MM_SHUFFLE(3, 0, 2, 1));

            // abcz1_0 = [b0*gamma,
            //            c0*alpha,
            //            a0*betta,
            //            0,
            //            ... ]
            __m512 abcz1_0 = _mm512_mul_ps(cGabz, bcaz0);
            // abcz1_1 = [c0*alpha - b0*gamma,
            //            a0*betta - c0*alpha,
            //            b0*gamma - a0*betta,
            //            0,
            //            ... ]
            __m512 abcz1_1 = _mm512_fmsub_ps(cAbgz, cabz0, abcz1_0);
            // abcz1_2 = [a0 + a0*(c0*alpha - b0*gamma),
            //            b0 + b0*(a0*betta - c0*alpha),
            //            c0 + c0*(b0*gamma - a0*betta),
            //            0,
            //            ... ]
            __m512 abcz1_2 = _mm512_fmadd_ps(abcz0, abcz1_1, abcz0);
            // abcz1 = [Max(Min(a0 + a0*(c0*alpha - b0*gamma), 1), 0),
            //          Max(Min(b0 + b0*(a0*betta - c0*alpha), 1), 0),
            //          Max(Min(c0 + c0*(b0*gamma - a0*betta), 1), 0),
            //          0,
            //          ... ]
            __m512 abcz1_3 = _mm512_max_ps(abcz1_2, cMax);
            __m512 abcz1 = _mm512_min_ps(abcz1_3, cMin);
            // unrolled x+4
            __m512 abcz1_0_4x = _mm512_mul_ps(cGabz, bcaz0_4x);
            __m512 abcz1_1_4x = _mm512_fmsub_ps(cAbgz, cabz0_4x, abcz1_0_4x);
            __m512 abcz1_2_4x = _mm512_fmadd_ps(abcz0_4x, abcz1_1_4x, abcz0_4x);
            __m512 abcz1_3_4x = _mm512_max_ps(abcz1_2_4x, cMax);
            __m512 abcz1_4x = _mm512_min_ps(abcz1_3_4x, cMin);
            // unrolled x+8
            __m512 abcz1_0_8x = _mm512_mul_ps(cGabz, bcaz0_8x);
            __m512 abcz1_1_8x = _mm512_fmsub_ps(cAbgz, cabz0_8x, abcz1_0_8x);
            __m512 abcz1_2_8x = _mm512_fmadd_ps(abcz0_8x, abcz1_1_8x, abcz0_8x);
            __m512 abcz1_3_8x = _mm512_max_ps(abcz1_2_8x, cMax);
            __m512 abcz1_8x = _mm512_min_ps(abcz1_3_8x, cMin);
            // unrolled x+12
            __m512 abcz1_0_12x = _mm512_mul_ps(cGabz, bcaz0_12x);
            __m512 abcz1_1_12x = _mm512_fmsub_ps(cAbgz, cabz0_12x, abcz1_0_12x);
            __m512 abcz1_2_12x = _mm512_fmadd_ps(abcz0_12x, abcz1_1_12x, abcz0_12x);
            __m512 abcz1_3_12x = _mm512_max_ps(abcz1_2_12x, cMax);
            __m512 abcz1_12x = _mm512_min_ps(abcz1_3_12x, cMin);

            // rgba_2 = [r0, g0, b0, a0,
            //           r1, g1, b1, a1,
            //           r2, g2, b2, a2,
            //           r3, g3, b3, a3]
            __m512 rgba_0 = _mm512_fmadd_ps(abcz1, cRgbaMult, cRgbaAdd);
            __m512i rgba_1 = _mm512_cvtps_epu32(rgba_0);
            __m512i rgba_2 = _mm512_sllv_epi32(rgba_1, cRgbaLeftShiftCount);
            // rgba_3 = [r0, g0, r0, g0,
            //           r1, g1, r1, g1,
            //           r2, g2, r2, g2,
            //           r3, g3, r3, g3]
            __m512i rgba_3 = _mm512_unpacklo_epi64(rgba_2, rgba_2);
            // rgba_4 = [-, -, r0|b0, g0|a0,
            //           -, -, r1|b1, g1|a1,
            //           -, -, r2|b2, g2|a2,
            //           -, -, r3|b3, g3|a3]
            __m512i rgba_4 = _mm512_or_epi32(rgba_2, rgba_3);
            // rgba_5 = [-, -, -, r0|b0,
            //           -, -, -, r1|b1,
            //           -, -, -, r2|b2,
            //           -, -, -, r3|b3]
            __m512i rgba_5 = _mm512_shuffle_epi32(rgba_4, _MM_SHUFFLE(2, 2, 2, 2));
            // rgba_6 = [-, -, -, r0|g0|b0|a0,
            //           -, -, -, r1|g1|b1|a1,
            //           -, -, -, r2|g2|b2|a2,
            //           -, -, -, r3|g3|b3|a3]
            __m512i rgba_6 = _mm512_or_epi32(rgba_4, rgba_5);
            // rgba_7 = [r0|g0|b0|a0, r1|g1|b1|a1, r2|g2|b2|a2, r3|g3|b3|a3,
            //                     -,           -,           -,           -,
            //                     -,           -,           -,           -,
            //                     -,           -,           -,           -]
            __m512i rgba_7 = _mm512_maskz_permutexvar_epi32((__mmask16)0b0000000000001111, cRgbaPermuteIdx, rgba_6);
            // unrolled x+4
            __m512 rgba_0_4x = _mm512_fmadd_ps(abcz1_4x, cRgbaMult, cRgbaAdd);
            __m512i rgba_1_4x = _mm512_cvtps_epu32(rgba_0_4x);
            __m512i rgba_2_4x = _mm512_sllv_epi32(rgba_1_4x, cRgbaLeftShiftCount);
            __m512i rgba_3_4x = _mm512_unpacklo_epi64(rgba_2_4x, rgba_2_4x);
            __m512i rgba_4_4x = _mm512_or_epi32(rgba_2_4x, rgba_3_4x);
            __m512i rgba_5_4x = _mm512_shuffle_epi32(rgba_4_4x, _MM_SHUFFLE(2, 2, 2, 2));
            __m512i rgba_6_4x = _mm512_or_epi32(rgba_4_4x, rgba_5_4x);
            __m512i rgba_7_4x = _mm512_maskz_permutexvar_epi32((__mmask16)0b0000000011110000, cRgbaPermuteIdx_4x, rgba_6_4x);
            // unrolled x+8
            __m512 rgba_0_8x = _mm512_fmadd_ps(abcz1_8x, cRgbaMult, cRgbaAdd);
            __m512i rgba_1_8x = _mm512_cvtps_epu32(rgba_0_8x);
            __m512i rgba_2_8x = _mm512_sllv_epi32(rgba_1_8x, cRgbaLeftShiftCount);
            __m512i rgba_3_8x = _mm512_unpacklo_epi64(rgba_2_8x, rgba_2_8x);
            __m512i rgba_4_8x = _mm512_or_epi32(rgba_2_8x, rgba_3_8x);
            __m512i rgba_5_8x = _mm512_shuffle_epi32(rgba_4_8x, _MM_SHUFFLE(2, 2, 2, 2));
            __m512i rgba_6_8x = _mm512_or_epi32(rgba_4_8x, rgba_5_8x);
            __m512i rgba_7_8x = _mm512_maskz_permutexvar_epi32((__mmask16)0b0000111100000000, cRgbaPermuteIdx_8x, rgba_6_8x);
            // unrolled x+12
            __m512 rgba_0_12x = _mm512_fmadd_ps(abcz1_12x, cRgbaMult, cRgbaAdd);
            __m512i rgba_1_12x = _mm512_cvtps_epu32(rgba_0_12x);
            __m512i rgba_2_12x = _mm512_sllv_epi32(rgba_1_12x, cRgbaLeftShiftCount);
            __m512i rgba_3_12x = _mm512_unpacklo_epi64(rgba_2_12x, rgba_2_12x);
            __m512i rgba_4_12x = _mm512_or_epi32(rgba_2_12x, rgba_3_12x);
            __m512i rgba_5_12x = _mm512_shuffle_epi32(rgba_4_12x, _MM_SHUFFLE(2, 2, 2, 2));
            __m512i rgba_6_12x = _mm512_or_epi32(rgba_4_12x, rgba_5_12x);
            __m512i rgba_7_12x = _mm512_maskz_permutexvar_epi32((__mmask16)0b1111000000000000, cRgbaPermuteIdx_12x, rgba_6_12x);

            __m512i rgbaSum0 = _mm512_add_epi32(rgba_7, rgba_7_4x);
            __m512i rgbaSum1 = _mm512_add_epi32(rgba_7_8x, rgba_7_12x);
            __m512i rgba = _mm512_add_epi32(rgbaSum0, rgbaSum1);

            _mm512_storeu_epi32(pixels + (y-1)*width + (x-1), rgba);

            _mm512_storeu_ps(ABCZ1 + (y*width + x)*4, abcz1);
            // unrolled x+4
            _mm512_storeu_ps(ABCZ1 + (y*width + x+4)*4, abcz1_4x);
            // unrolled x+8
            _mm512_storeu_ps(ABCZ1 + (y*width + x+8)*4, abcz1_8x);
            // unrolled x+12
            _mm512_storeu_ps(ABCZ1 + (y*width + x+12)*4, abcz1_12x);
        }
    }
    swap(&bz->ABCZ0, &bz->ABCZ1);
}

internal void
bzsim_free(BZSim *bz)
{
    free(bz->ABCZ0);
    free(bz->ABCZ1);
}
