#include <stdlib.h>
#include "utility.h"

f32 random_f32() {
  return rand() / (RAND_MAX + 1.0f);
}

f32 random_f32_bound(f32 min, f32 max) {
  return min + (max-min)*random_f32();
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

