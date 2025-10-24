#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

#include <time.h>

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
void pixels_to_ppm(vec3 *pixels_colors, u32 pixels_width, u32 pixels_height) {
  FILE *img_file = fopen("image.ppm", "wb");
  if (img_file == NULL) {
    fprintf(stderr, "Couldn't open the file for writing\n");
    return;
  }

  u64 written = fwrite("P6\n", sizeof(char), strlen("P6\n"), img_file);
  
  char pixels_width_buf[10];
  i32 pixels_width_digits = char_buf_to_uint32(pixels_width_buf, 10, pixels_width);
  written = fwrite(pixels_width_buf, sizeof(char), pixels_width_digits, img_file);

  written = fwrite(" ", sizeof(char), 1, img_file);

  char pixels_height_buf[10];
  i32 pixels_height_digits = char_buf_to_uint32(pixels_height_buf, 10, pixels_height);
  written = fwrite(pixels_height_buf, sizeof(char), pixels_height_digits, img_file);

  written = fwrite("\n255\n", sizeof(char), strlen("\n255\n"), img_file);

  for (i32 i = 0; i < pixels_height; ++i) {
    for (i32 j = 0; j < pixels_width; ++j) {
      u8 r = (u8)(pixels_colors[i * pixels_width + j].x * 255);
      u8 g = (u8)(pixels_colors[i * pixels_width + j].y * 255);
      u8 b = (u8)(pixels_colors[i * pixels_width + j].z * 255);
      // i32 bgr = (b << 16) | (g << 8) | r;
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

b32 vec3_is_same_approx(vec3 v1, vec3 v2) {
  return (v1.x - v2.x < 0.000001f) 
    && (v1.y - v2.y < 0.000001f)  
    && (v1.z - v2.z < 0.000001f);
}

vec3 Color(f32 x, f32 y, f32 z) {
  vec3 result = { x, y, z };
  return result;
}

f32 hit_sphere(vec3 sphere_loc, float radius, vec3 ray_origin, vec3 ray_direction) {
  vec3 oc = vec3_sub(sphere_loc, ray_origin);
  float a = vec3_dot_prod(ray_direction, ray_direction);
  float b = -2.0f * vec3_dot_prod(ray_direction, oc);
  float c = vec3_dot_prod(oc, oc) - radius * radius;
  float discriminant = b * b - 4*a*c;

  if (discriminant < 0)
    return -1.0f;
  else
    return (-b - sqrtf(discriminant)) / (2.0f*a);
}

f32 hit_sphere_slower_simplified(vec3 sphere_loc, float radius, vec3 ray_origin, vec3 ray_direction) {
  vec3 oc = vec3_sub(sphere_loc, ray_origin);
  float direction_len = vec3_length(ray_direction);
  float a = direction_len * direction_len;
  float h = vec3_dot_prod(ray_direction, oc);
  float oc_len = vec3_length(oc);
  float c = oc_len * oc_len - radius * radius;
  float discriminant = h*h - a * c;

  if (discriminant < 0)
    return -1.0f;
  else
    return (h - sqrtf(discriminant)) / a;
}

vec3 ray_at(vec3 origin, float t, vec3 direction) {
  return vec3_add(origin, vec3_scale(t, direction));
}

vec3 ray_color(vec3 origin, vec3 direction) {
  float t = hit_sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, origin, direction);
  if (t > 0.0f) {
    vec3 normal = vec3_to_unit_vec(
        vec3_sub(ray_at(origin, t, direction), Vec3(0.0f, 0.0f, -1.0f)));
    return vec3_scale(0.5f, Color(normal.x + 1, normal.y + 1, normal.z + 1));
  }

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
  fprintf(stderr, "vec3_to_unit test start\n");
  vec3 inputs[] = { Vec3(1.0f, 1.0f, 1.0f), Vec3(0.5f, 0.5f, 0.5f),
    Vec3(10.0f, 0.5f, 12.7f) };
  vec3 expected[] = { Vec3(0.57735f, 0.57735f, 0.57735f),
    Vec3(0.57735f, 0.57735f, 0.57735f), Vec3(0.61835f, 0.030917f, 0.7853f) };
  i32 vec_count = sizeof(inputs) / sizeof(inputs[0]);
  for (i32 i = 0; i < vec_count; ++i) {
    inputs[i] = vec3_to_unit_vec(inputs[i]);
    if (!vec3_is_same_approx(inputs[i], expected[i])) {
      fprintf(stderr, "vec3_to_unit test failure\n");
      fprintf(stderr, "Failed at input vector #%d\n", i);
      fprintf(stderr, "Expected (%f, %f, %f), got (%f, %f, %f)", 
          expected[i].x, expected[i].y, expected[i].z,
          inputs[i].x, inputs[i].y, inputs[i].z);
      return 0;
    }
  }
  fprintf(stderr, "vec3_to_unit test success\n");
  return 1;
}

i64 timer_start_ns() {
  struct timespec time;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time);
  return time.tv_sec * 1e6 + time.tv_nsec;
}

i64 timer_stop_ns(i64 start_time) {
  struct timespec time;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time);
  return time.tv_sec * 1e6 + time.tv_nsec - start_time;
}

f64 timer_start_ms() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (f64)time.tv_sec * 1e3 + (f64)time.tv_nsec / 1e6;
}

f64 timer_stop_ms(f64 start_time) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (f64)time.tv_sec * 1e3 + (f64)time.tv_nsec / 1e6 - start_time;
}

i32 main() {
  vec3_to_unit_vec_test();

	i32 img_width = 400;
	f32 img_ratio = 16.0f/9.0f;
	i32 img_height = (int)(img_width / img_ratio);
	
	f32 camera_viewport_dist = 1.0f;
	f32 viewport_height = 2.0f;
	f32 viewport_ratio = (f32)img_width /  (f32)img_height;
	f32 viewport_width = viewport_height * viewport_ratio;
	vec3 camera_pos = { 0.0f, 0.0f, 0.0f };
	
	vec3 viewport_left_upper_corner = camera_pos;
	viewport_left_upper_corner.x -= viewport_width / 2;
	viewport_left_upper_corner.y += viewport_height / 2;
	viewport_left_upper_corner.z -= camera_viewport_dist;

	f32 pixel_dist_y = viewport_height / img_height;
	f32 pixel_dist_x = viewport_width / img_width;
	
	vec3 current_pixel_center = viewport_left_upper_corner;
	current_pixel_center.x += 0.5 * pixel_dist_x;
	current_pixel_center.y -= 0.5 * pixel_dist_y;
	
  vec3 *rays_directions = calloc(img_height * img_width, sizeof(vec3));
  vec3 *pixels_colors = calloc(img_height * img_width, sizeof(vec3));

  f32 p00x = current_pixel_center.x;
	
  f64 time_start = timer_start_ms();
	for (i32 i = 0; i < img_height; ++i) {
		for (i32 j = 0; j < img_width; ++j) {
      u64 curr_idx = i * img_width + j;
			rays_directions[curr_idx] = vec3_sub(current_pixel_center, camera_pos);

      pixels_colors[curr_idx] = ray_color(camera_pos, rays_directions[curr_idx]);
      current_pixel_center.x += pixel_dist_x;
		}
    current_pixel_center.y -= pixel_dist_y;
    current_pixel_center.x = p00x;
	}
  f64 time_elapsed = timer_stop_ms(time_start);
  printf("Rendering time: %fms\n", time_elapsed);

  pixels_to_ppm(pixels_colors, img_width, img_height);
}
