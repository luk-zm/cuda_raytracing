#ifndef VEC3_H
#define VEC3_H

typedef struct {
	f32 x;
	f32 y;
	f32 z;
} vec3;

__host__ __device__
inline vec3 Vec3(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

__host__ __device__
inline vec3 Color(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

__host__ __device__
inline vec3 vec3_add(vec3 v1, vec3 v2) {
  vec3 result = {0};
  result.x = v1.x + v2.x;
  result.y = v1.y + v2.y;
  result.z = v1.z + v2.z;
  return result;
}

__host__ __device__
inline vec3 vec3_sub(vec3 v1, vec3 v2) {
  vec3 result = {0};
  result.x = v1.x - v2.x;
  result.y = v1.y - v2.y;
  result.z = v1.z - v2.z;
  return result;
}

__host__ __device__
inline vec3 vec3_scale(f32 scalar, vec3 v) {
  vec3 result = {0};
  result.x = v.x * scalar;
  result.y = v.y * scalar;
  result.z = v.z * scalar;
  return result;
}

__host__ __device__
inline vec3 vec3_sign_flip(vec3 v) {
  vec3 result = { -v.x, -v.y, -v.z };
  return result;
}

__host__ __device__
inline vec3 *vec3_inplace_add(vec3 *v1, vec3 v2) {
  v1->x += v2.x;
  v1->y += v2.y;
  v1->z += v2.z;
  return v1;
}

__host__ __device__
inline vec3 *vec3_inplace_sub(vec3 *v1, vec3 v2) {
  v1->x -= v2.x;
  v1->y -= v2.y;
  v1->z -= v2.z;
  return v1;
}

__host__ __device__
inline vec3 *vec3_inplace_scale(f32 scalar, vec3* v) {
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

__host__ __device__
inline f32 vec3_dot_prod(vec3 v1, vec3 v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__
inline vec3 vec3_cross_product(vec3 v1, vec3 v2) {
  vec3 result = {0};
  result.x = v1.y * v2.z - v1.z * v2.y;
  result.y = v1.z * v2.x - v1.x * v2.z;
  result.z = v1.x * v2.y - v1.y * v2.x;
  return result;
}

__host__ __device__
inline f32 vec3_length(vec3 v) {
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

__host__ __device__
inline vec3 vec3_to_unit_vec(vec3 v) {
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

inline vec3 vec3_random() {
  return Vec3(random_f32(), random_f32(), random_f32());
}

inline vec3 vec3_random_bound(f32 min, f32 max) {
  return Vec3(random_f32_bound(min, max), random_f32_bound(min, max), random_f32_bound(min, max));
}

inline vec3 vec3_random_unit_vector() {
  while (1) {
    vec3 rand = vec3_random_bound(-1.0f, 1.0f);
    f32 rand_len = vec3_length(rand);
    f32 rand_len_squared = rand_len * rand_len;
    if (1e-50 < rand_len_squared && rand_len_squared <= 1.0f)
      return vec3_scale(1.0f/rand_len, rand);
  }
}

__device__
inline vec3 cuda_vec3_random(curandState *rng) {
  return Vec3(curand_uniform(rng), curand_uniform(rng), curand_uniform(rng));
}

__device__
inline vec3 cuda_vec3_random_bound(curandState *rng, f32 min, f32 max) {
  return Vec3(cuda_random_f32_bound(rng, min, max),
      cuda_random_f32_bound(rng, min, max),
      cuda_random_f32_bound(rng, min, max));
}

__device__
inline vec3 cuda_vec3_random_unit_vector(curandState *rng) {
  while (1) {
    vec3 rand = cuda_vec3_random_bound(rng, -1.0f, 1.0f);
    f32 rand_len = vec3_length(rand);
    f32 rand_len_squared = rand_len * rand_len;
    if (1e-50 < rand_len_squared && rand_len_squared <= 1.0f)
      return vec3_scale(1.0f/rand_len, rand);
  }
}
vec3 vec3_random_on_hemisphere(vec3 normal) {
  vec3 on_unit_sphere = vec3_random_unit_vector();
  if (vec3_dot_prod(on_unit_sphere, normal) > 0.0f)
    return on_unit_sphere;
  else
    return vec3_sign_flip(on_unit_sphere);
}

__host__ __device__
inline vec3 vec3_comp_scale(vec3 v1, vec3 v2) {
  vec3 result = Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
  return result;
}

__host__ __device__
inline b32 vec3_is_near_zero(vec3 v) {
  f32 s = 1e-8;
  return fabsf(v.x) < s && fabsf(v.y) < s && fabsf(v.z) < s;
}

__host__ __device__
inline vec3 line_at(vec3 origin, f32 t, vec3 direction) {
  return vec3_add(origin, vec3_scale(t, direction));
}

#endif
