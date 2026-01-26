#ifndef UTILITY_H
#define UTILITY_H

#include <stdint.h>

#define MIN(a,b) a < b ? a : b
#define MAX(a,b) a < b ? b : a

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef int32_t b32; // boolean
typedef float f32;
typedef double f64;

f32 random_f32();
f32 random_f32_bound(f32 min, f32 max);
f32 random_f32_bound(f32 min, f32 max);
i32 random_i32_bound(i32 min, i32 max);
f32 clampf(f32 val, f32 min, f32 max);
i32 char_buf_to_uint32(char *buf, i32 buf_length, u32 num);
b32 interval_overlap(f32 x0, f32 x1, f32 y0, f32 y1);
void interval_sort(f32 *x0, f32 *x1);
vec3 line_at(vec3 origin, f32 t, vec3 direction);

#endif
