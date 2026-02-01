#include <stdlib.h>
#include "utility.h"

f32 random_f32() {
  return rand() / (RAND_MAX + 1.0f);
}

f32 random_f32_bound(f32 min, f32 max) {
  return min + (max-min)*random_f32();
}

i32 random_i32_bound(i32 min, i32 max) {
  return (int)(random_f32_bound(min, max + 1));
}

f32 clampf(f32 val, f32 min, f32 max) {
  if (val < min)
    return min;
  if (val > max)
    return max;
  return val;
}

i32 char_buf_to_uint32(char *buf, i32 buf_length, u32 num) {
  if (buf_length <= 0)
    return 0;

  buf[0] = '0' + num % 10;
  num /= 10;
  i32 i;
  for (i = 1; i < buf_length && num != 0; ++i) {
    buf[i] = '0' + num % 10;
    num /= 10;
  }

  i32 number_of_digits = i;
  for (i = 0; i < number_of_digits / 2; ++i) {
    char temp = buf[i];
    buf[i] = buf[number_of_digits - i - 1];
    buf[number_of_digits - i - 1] = temp;
  }

  return number_of_digits;
}

b32 interval_overlap(f32 x0, f32 x1, f32 y0, f32 y1) {
  f32 shared_min = (x0 < y0 ? y0 : x0);
  f32 shared_max = (x1 < y1 ? x1 : y1);
  return shared_min < shared_max;
}

void interval_expand(f32 *min, f32 *max, f32 delta) {
  f32 padding = delta / 2;
  *min -= padding;
  *max += padding;
}

void interval_sort(f32 *x0, f32 *x1) {
  if (*x0 > *x1) {
    f32 temp = *x0;
    *x0 = *x1;
    *x1 = temp;
  }
}

vec3 line_at(vec3 origin, f32 t, vec3 direction) {
  return vec3_add(origin, vec3_scale(t, direction));
}

f32 degrees_to_radians(f32 angle) {
  f32 result = angle * (pi / 180);
  return result;
}
