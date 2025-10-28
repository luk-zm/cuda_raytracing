#include <math.h>
 
#include "vec3.h"

vec3 Vec3(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

vec3 Color(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

vec3 vec3_add(vec3 v1, vec3 v2) {
  vec3 result = {0};
  result.x = v1.x + v2.x;
  result.y = v1.y + v2.y;
  result.z = v1.z + v2.z;
  return result;
}

vec3 vec3_sub(vec3 v1, vec3 v2) {
  vec3 result = {0};
  result.x = v1.x - v2.x;
  result.y = v1.y - v2.y;
  result.z = v1.z - v2.z;
  return result;
}

vec3 vec3_scale(f32 scalar, vec3 v) {
  vec3 result = {0};
  result.x = v.x * scalar;
  result.y = v.y * scalar;
  result.z = v.z * scalar;
  return result;
}

vec3 vec3_sign_flip(vec3 v) {
  vec3 result = { -v.x, -v.y, -v.z };
  return result;
}

vec3 *vec3_inplace_add(vec3 *v1, vec3 v2) {
  v1->x += v2.x;
  v1->y += v2.y;
  v1->z += v2.z;
  return v1;
}

vec3 *vec3_inplace_sub(vec3 *v1, vec3 v2) {
  v1->x -= v2.x;
  v1->y -= v2.y;
  v1->z -= v2.z;
  return v1;
}

vec3 *vec3_inplace_scale(f32 scalar, vec3* v) {
  v->x *= scalar;
  v->y *= scalar;
  v->z *= scalar;
  return v;
}

vec3 *vec3_inplace_sign_flip(vec3 *v) {
  v->x = -v->x;
  v->y = -v->y;
  v->z = -v->z;
  return v;
}

f32 vec3_dot_prod(vec3 v1, vec3 v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

vec3 vec3_cross_product(vec3 v1, vec3 v2) {
  vec3 result = {0};
  result.x = v1.y * v2.z - v1.z * v2.y;
  result.y = v1.z * v2.x - v1.x * v2.z;
  result.z = v1.x * v2.y - v1.y * v2.x;
  return result;
}

f32 vec3_length(vec3 v) {
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

vec3 vec3_to_unit_vec(vec3 v) {
	f32 v_len = vec3_length(v);
  if (v_len == 0.0f) {
    return Vec3(0.0f, 0.0f, 0.0f);
  }

	vec3 result = {v.x / v_len, v.y / v_len, v.z / v_len};
	return result;
}

b32 vec3_is_same_approx(vec3 v1, vec3 v2) {
  return (v1.x - v2.x < 0.000001f) 
    && (v1.y - v2.y < 0.000001f)  
    && (v1.z - v2.z < 0.000001f);
}

