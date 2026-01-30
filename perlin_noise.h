#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#define PERLIN_NOISE_POINTS_COUNT 256
typedef struct {
  vec3 random_vec3s[PERLIN_NOISE_POINTS_COUNT];
  i32 perm_x[PERLIN_NOISE_POINTS_COUNT];
  i32 perm_y[PERLIN_NOISE_POINTS_COUNT];
  i32 perm_z[PERLIN_NOISE_POINTS_COUNT];
} PerlinNoise;

f32 perlin_noise_val(PerlinNoise *noise, vec3 point);
void perlin_noise_gen_permutation(i32 *arr, i32 count);
PerlinNoise make_perlin_noise();
f32 perlin_turbulence(PerlinNoise *noise, vec3 point, i32 depth);

#endif
