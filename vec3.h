#ifndef VEC3_H
#define VEC3_H

#include <stdint.h>

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

typedef struct {
	f32 x;
	f32 y;
	f32 z;
} vec3;

vec3 Vec3(f32 x, f32 y, f32 z);
vec3 Color(f32 x, f32 y, f32 z);
vec3 vec3_add(vec3 v1, vec3 v2);
vec3 vec3_sub(vec3 v1, vec3 v2);
vec3 vec3_scale(f32 scalar, vec3 v);
vec3 vec3_sign_flip(vec3 v);
vec3 *vec3_inplace_add(vec3 *v1, vec3 v2);
vec3 *vec3_inplace_sub(vec3 *v1, vec3 v2);
vec3 *vec3_inplace_scale(f32 scalar, vec3* v);
vec3 *vec3_inplace_sign_flip(vec3 *v);
f32 vec3_dot_prod(vec3 v1, vec3 v2);
vec3 vec3_cross_product(vec3 v1, vec3 v2);
f32 vec3_length(vec3 v);
vec3 vec3_to_unit_vec(vec3 v);
b32 vec3_is_same_approx(vec3 v1, vec3 v2);
vec3 vec3_random();
vec3 vec3_random_bound(f32 min, f32 max);
vec3 vec3_random_unit_vector();
vec3 vec3_random_on_hemisphere(vec3 normal);

#endif
