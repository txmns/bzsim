#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint8_t   U8;
typedef uint16_t  U16;
typedef uint32_t  U32;
typedef uint64_t  U64;
typedef int8_t    S8;
typedef int16_t   S16;
typedef int32_t   S32;
typedef int64_t   S64;
typedef S32       B32;
typedef float     F32;
typedef double    F64;

#define True  (B32)1
#define False (B32)0

#define internal      static
#define global        static
#define local_persist static

#define Trap __builtin_trap()

#if BUILD_DEBUG
#define Assert(x)   \
    do {            \
        if (!(x)) { \
            Trap;   \
        }           \
    } while (0)
#else
#define Assert(x) (void)(x)
#endif

#define ArrayCap(a) (sizeof(a) / sizeof((a)[0]))

#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Max(a, b) ((a) > (b) ? (a) : (b))

internal F32
rand_uniform(void)
{
    return ((F32)rand() / (F32)RAND_MAX);
}

internal void
swap(F32 **a, F32 **b)
{
    F32 *tmp = *a;
    *a = *b;
    *b = tmp;
}
