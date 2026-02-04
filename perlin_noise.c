__host__ __device__
inline f32 perlin_interp(vec3 c[2][2][2], f32 u, f32 v, f32 w) {
  f32 uu = u*u*(3 - 2*u);
  f32 vv = v*v*(3 - 2*v);
  f32 ww = w*w*(3 - 2*w);
  f32 accum = 0.0f;

  for (i32 i=0; i < 2; i++) {
    for (i32 j=0; j < 2; j++) {
      for (i32 k=0; k < 2; k++) {
        vec3 weight_v = { u - (f32)i, v - (f32)j, w - (f32)k };
        accum += (i*uu + (1-i)*(1-uu))
               * (j*vv + (1-j)*(1-vv))
               * (k*ww + (1-k)*(1-ww))
               * vec3_dot_prod(c[i][j][k], weight_v);
      }
    }
  }
  return accum;
}

__host__ __device__
inline f32 perlin_noise_val(PerlinNoise *noise, vec3 point) {
  f32 u = point.x - floorf(point.x);
  f32 v = point.y - floorf(point.y);
  f32 w = point.z - floorf(point.z);

  i32 i = (i32)floorf(point.x);
  i32 j = (i32)floorf(point.y);
  i32 k = (i32)floorf(point.z);
  vec3 c[2][2][2];

  for (i32 di = 0; di < 2; ++di) {
    for (i32 dj = 0; dj < 2; ++dj) {
      for (i32 dk = 0; dk < 2; ++dk) {
        c[di][dj][dk] = noise->random_vec3s[
          noise->perm_x[(i+di) & 255] ^
          noise->perm_x[(j+dj) & 255] ^
          noise->perm_x[(k+dk) & 255]
        ];
      }
    }
  }

  return perlin_interp(c, u, v, w);
}

void perlin_noise_gen_permutation(i32 *arr, i32 count) {
  for (i32 i = 0; i < count; ++i) {
    arr[i] = i;
  }

  for (i32 i = count - 1; i > 0; --i) {
    i32 target = random_i32_bound(0, i);
    i32 tmp = arr[i];
    arr[i] = arr[target];
    arr[target] = tmp;
  }
}

PerlinNoise make_perlin_noise() {
  PerlinNoise result = {0};
  for (i32 i = 0; i < PERLIN_NOISE_POINTS_COUNT; ++i) {
    result.random_vec3s[i] = vec3_to_unit_vec(vec3_random_bound(-1, 1));
  }
  perlin_noise_gen_permutation(result.perm_x, PERLIN_NOISE_POINTS_COUNT);
  perlin_noise_gen_permutation(result.perm_y, PERLIN_NOISE_POINTS_COUNT);
  perlin_noise_gen_permutation(result.perm_z, PERLIN_NOISE_POINTS_COUNT);
  return result;
}

__host__ __device__
inline f32 perlin_turbulence(PerlinNoise *noise, vec3 point, i32 depth) {
  f32 accum = 0.0f;
  vec3 tmp_p = point;
  f32 weight = 1.0f;

  for (i32 i = 0; i < depth; ++i) {
    accum += weight * perlin_noise_val(noise, tmp_p);
    weight *= 0.5f;
    vec3_inplace_scale(2, &tmp_p);
  }

  return fabsf(accum);
}
