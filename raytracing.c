#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef int32_t b32; // boolean
typedef float f32;
typedef double f64;

s32 char_buf_to_uint32(char *buf, s32 buf_length, u32 num) {
  if (buf_length <= 0)
    return 0;

  buf[0] = '0' + num % 10;
  num /= 10;
  s32 i;
  for (i = 1; i < buf_length && num != 0; ++i) {
    buf[i] = '0' + num % 10;
    num /= 10;
  }

  s32 number_of_digits = i;
  for (i = 0; i < number_of_digits / 2; ++i) {
    char temp = buf[i];
    buf[i] = buf[number_of_digits - i - 1];
    buf[number_of_digits - i - 1] = temp;
  }

  return number_of_digits;
}

typedef struct {
	f32 x;
	f32 y;
	f32 z;
} vec3;

vec3 Vec3(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

// TODO: handle fwrite return values, consider s_fopen
void pixels_to_ppm(vec3 *pixels_colors, u32 pixels_height, u32 pixels_width) {
  FILE *img_file = fopen("image.ppm", "wb");
  fwrite("P6\n", sizeof(char), strlen("P6\n"), img_file);
  
  char pixels_width_buf[10];
  s32 pixels_width_digits = char_buf_to_uint32(pixels_width_buf, 10, pixels_width);
  fwrite(pixels_width_buf, sizeof(char), pixels_width_digits, img_file);

  fwrite(" ", sizeof(char), 1, img_file);

  char pixels_height_buf[10];
  s32 pixels_height_digits = char_buf_to_uint32(pixels_height_buf, 10, pixels_height);
  fwrite(pixels_height_buf, sizeof(char), pixels_height_digits, img_file);

  fwrite("\n255\n", sizeof(char), strlen("\n255\n"), img_file);

  for (s32 i = 0; i < pixels_height; ++i) {
    for (s32 j = 0; j < pixels_width; ++j) {
      u8 r = (u8)(pixels_colors[i * pixels_width + j].x * 255);
      u8 g = (u8)(pixels_colors[i * pixels_width + j].y * 255);
      u8 b = (u8)(pixels_colors[i * pixels_width + j].z * 255);
      // s32 bgr = (b << 16) | (g << 8) | r;
      fwrite(&r, sizeof(r), 1, img_file);
      fwrite(&g, sizeof(g), 1, img_file);
      fwrite(&b, sizeof(b), 1, img_file);
    }
  }

  fclose(img_file);
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

b32 vec3_is_same(vec3 v1, vec3 v2) {
  return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

vec3 Color(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

vec3 ray_color(vec3 origin, vec3 direction) {
  vec3 unit_direction = vec3_to_unit_vec(direction);
  f32 blend_percent = 0.5f * (unit_direction.y + 1.0f);
  return vec3_add(vec3_scale(1.0f - blend_percent, Color(1.0f, 1.0f, 1.0f)),
      vec3_scale(blend_percent, Color(0.5f, 0.7f, 1.0f)));
}

#define ZERO_VEC Vec3(0.0f, 0.0f, 0.0f)

vec3 zero_vec() {
  return Vec3(0.0f, 0.0f, 0.0f);
}

b32 vec3_to_unit_vec_test() {
  vec3 input_vectors[] = { Vec3(1.0f, 1.0f, 1.0f), Vec3(0.5f, 0.5f, 0.5f),
    Vec3(10.0f, 0.5f, 12.7f) };
  vec3 output_vectors[] = { Vec3(0.57735f, 0.57735f, 0.57735f),
    Vec3(0.57735f, 0.57735f, 0.57735f), Vec3(0.61835f, 0.030917f, 0.7853f) };
  s32 vec_count = sizeof(input_vectors) / sizeof(input_vectors[0]);
  for (s32 i = 0; i < vec_count; ++i) {
    if (vec3_is_same(input_vectors[i], output_vectors[i])) {
      fprintf(stderr, "Failed at input vector #%d\n", i);
      return 0;
    }
  }
  return 1;
}

void ray_color_test() {
  vec3 directions[] = { Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f),
    Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f)};
}

s32 main() {
  vec3_to_unit_vec_test();

	s32 img_height = 400;
	f32 img_ratio = 16.0/9.0;
	s32 img_width = (int)(img_height * img_ratio);
	
	f32 camera_viewport_dist = 1.0f;
	f32 viewport_height = 2.0f;
	f32 viewport_ratio = (f32)img_height /  (f32)img_width;
	f32 viewport_width = viewport_height * viewport_ratio;
	vec3 camera_pos = { 0.0f, 0.0f, 0.0f };
	
	vec3 viewport_q = camera_pos;
	viewport_q.x -= viewport_width / 2;
	viewport_q.y += viewport_height / 2;
	viewport_q.z -= camera_viewport_dist;

	f32 pixel_dist = viewport_height / img_height;
	
	vec3 p00 = viewport_q;
	p00.x += pixel_dist;
	p00.y -= pixel_dist;
	
  // todo: get rid of VLAs
	vec3 rays_directions[img_height][img_width];
	vec3 pixels_colors[img_height][img_width];
	
	for (s32 i = 0; i < img_height; ++i) {
		for (s32 j = 0; j < img_width; ++j) {
			rays_directions[i][j].x = p00.x + i * pixel_dist;
			rays_directions[i][j].y = p00.y + j * pixel_dist;
      rays_directions[i][j].z = p00.z;
      vec3_inplace_sub(&rays_directions[i][j], camera_pos);

      pixels_colors[i][j] = ray_color(camera_pos, rays_directions[i][j]);
		}
	}

  pixels_to_ppm((vec3 *)pixels_colors, img_height, img_width);
}
