#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "vec3.h"
#include "utility.h"

#include "vec3.c" // unity build
#include "utility.c"

const float pi = 3.1415926535897932385f;

typedef struct {
  vec3 point_hit;
  vec3 normal;
  float t;
  b32 front_face;
} HitRecord;

// outward_normal is of unit length
void hit_record_set_face_normal(HitRecord *record, vec3 ray_direction, vec3 outward_normal) {
    record->front_face = vec3_dot_prod(ray_direction, outward_normal) < 0;
    record->normal = record->front_face ? outward_normal : vec3_sign_flip(outward_normal);
}

typedef enum {
  VIS_OBJECT_SPHERE
} VisObject;

typedef struct {
  vec3 center;
  f32 radius;
} Sphere;

Sphere make_sphere(vec3 center, f32 radius) {
  Sphere s = { center, radius };
  return s;
}

typedef struct {
  VisObject type;
  union {
    Sphere sphere;
  };
} Hittable;

typedef struct {
  Hittable *objects;
  u64 count;
} HittableList;

vec3 ray_at(vec3 origin, float t, vec3 direction) {
  return vec3_add(origin, vec3_scale(t, direction));
}

b32 hit(Hittable *hittable, float ray_tmin, float ray_tmax, vec3 ray_origin, vec3 ray_direction, HitRecord *record) {
  switch (hittable->type) {
    case VIS_OBJECT_SPHERE: {
      vec3 oc = vec3_sub(hittable->sphere.center, ray_origin);
      float a = vec3_dot_prod(ray_direction, ray_direction);
      float b = -2.0f * vec3_dot_prod(ray_direction, oc);
      float c = vec3_dot_prod(oc, oc) - hittable->sphere.radius * hittable->sphere.radius;
      float discriminant = b * b - 4*a*c;

      if (discriminant < 0)
        return 0;

      f32 sqrt_discriminant = sqrtf(discriminant);
      f32 root = (-b - sqrt_discriminant) / (2.0f*a);
      if (ray_tmin >= root || root >= ray_tmax) {
        root = (-b + sqrt_discriminant) / (2.0f*a);
        if (ray_tmin >= root || root >= ray_tmax) {
          return 0;
        }
      }

      record->t = root;
      record->point_hit = ray_at(ray_origin, root, ray_direction);
      vec3 outward_normal = vec3_scale(1.0f/hittable->sphere.radius,
          vec3_sub(record->point_hit, hittable->sphere.center));
      hit_record_set_face_normal(record, ray_direction, outward_normal);

      return 1;
    } break;
  }
  return 0;
}

b32 hittable_list_hit(HittableList list, vec3 ray_origin, vec3 ray_direction,
    double ray_tmin, double ray_tmax, HitRecord *record) {
  HitRecord temp_record;
  b32 hit_anything = 0;
  float closest_so_far = ray_tmax;

  for (int i = 0; i < list.count; ++i) {
    if (hit(&list.objects[i], ray_tmin, closest_so_far, ray_origin, ray_direction, &temp_record)) {
      hit_anything = 1;
      closest_so_far = temp_record.t;
      *record = temp_record;
    }
  }

  return hit_anything;
}

vec3 ray_color(vec3 origin, vec3 direction, i32 max_bounces, HittableList world) {
  if (max_bounces <= 0)
    return Color(0, 0, 0);

  HitRecord record = {0};
  if (hittable_list_hit(world, origin, direction, 0.001f, INFINITY, &record)) {
    vec3 direction = vec3_add(record.normal, vec3_random_unit_vector());
    return vec3_scale(0.5f, ray_color(record.point_hit, direction, max_bounces - 1, world));
  }

  vec3 unit_direction = vec3_to_unit_vec(direction);
  f32 blend_percent = 0.5f * (unit_direction.y + 1.0f);
  return vec3_add(vec3_scale(1.0f - blend_percent, Color(1.0f, 1.0f, 1.0f)),
      vec3_scale(blend_percent, Color(0.5f, 0.7f, 1.0f)));
}

f32 linear_to_gamma(f32 linear_component) {
  if (linear_component > 0)
    return sqrtf(linear_component);
  return 0.0f;
}

// TODO: handle fwrite return values, consider s_fopen
void pixels_to_ppm(vec3 *pixels_colors, u32 pixels_width, u32 pixels_height) {
  FILE *img_file = fopen("image.ppm", "wb");
  if (img_file == NULL) {
    fprintf(stderr, "Couldn't open the file for writing\n");
    return;
  }

  u64 written = fwrite("P6\n", sizeof(char), 3, img_file);
  
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
      f32 r = linear_to_gamma(pixels_colors[i * pixels_width + j].x);
      f32 g = linear_to_gamma(pixels_colors[i * pixels_width + j].y);
      f32 b = linear_to_gamma(pixels_colors[i * pixels_width + j].z);

      u8 rbyte = (u8)(256 * clampf(r, 0.0f, 0.999f));
      u8 gbyte = (u8)(256 * clampf(g, 0.0f, 0.999f));
      u8 bbyte = (u8)(256 * clampf(b, 0.0f, 0.999f));
      // i32 bgr = (b << 16) | (g << 8) | r;
      fwrite(&rbyte, sizeof(rbyte), 1, img_file);
      fwrite(&gbyte, sizeof(gbyte), 1, img_file);
      fwrite(&bbyte, sizeof(bbyte), 1, img_file);
    }
  }

  fclose(img_file);
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

#if LINUX
i64 timer_start_ns() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return time.tv_sec * 1000000 + time.tv_nsec;
}

i64 timer_stop_ns(i64 start_time) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return time.tv_sec * 1000000 + time.tv_nsec - start_time;
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

f64 timer_start() {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (f64)time.tv_sec + (f64)time.tv_nsec / 1000000000.0;
}

f64 timer_stop(f64 start_time) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return ((f64)time.tv_sec + (f64)time.tv_nsec / 1000000000.0) - start_time;
}

#elif WINDOWS
#include <intrin.h>
#include <windows.h>

LARGE_INTEGER freq;

f64 timer_start_ns() {
  return (f64)__rdtsc();
}

f64 timer_stop_ns(f64 start_time) {
  return (__rdtsc() - start_time) / 3.7;
}

f64 timer_start_ms() {
  return (f64)__rdtsc();
}

f64 timer_stop_ms(f64 start_time) {
  return ((f64)__rdtsc() - start_time) / 3700000.0;
}

u64 timer_start() {
  QueryPerformanceFrequency(&freq);
  LARGE_INTEGER ticks;
  QueryPerformanceCounter(&ticks);
  return ticks.QuadPart;
}

f64 timer_stop(u64 start_time) {
  LARGE_INTEGER ticks;
  QueryPerformanceCounter(&ticks);
  return (f64)(ticks.QuadPart - start_time) / (f64)freq.QuadPart;
}

#endif

i32 main() {
  vec3_to_unit_vec_test();

  HittableList world = {0};
  Hittable hittables[2] = { { VIS_OBJECT_SPHERE, make_sphere(Vec3(0, 0, -1), 0.5f) }, 
                            { VIS_OBJECT_SPHERE, make_sphere(Vec3(0, -100.5f, -1), 100.0f)} };

  world.objects = hittables;
  world.count = 2;

  i32 samples_per_pixel = 100;
  f32 pixel_samples_scale = 1.0f / (f32)samples_per_pixel;
  i32 max_bounces = 50;

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
	
  vec3 current_ray_direction = {0};
  vec3 *pixels_colors = calloc(img_height * img_width, sizeof(vec3));

  f32 p00x = current_pixel_center.x;
	
  f64 time_start = timer_start_ms();
  f64 time_start_s = timer_start();
	for (i32 i = 0; i < img_height; ++i) {
		for (i32 j = 0; j < img_width; ++j) {
      u64 curr_idx = i * img_width + j;
      for (i32 sample_num = 0; sample_num < samples_per_pixel; ++sample_num) {
        f32 x_offset = random_f32() - 0.5f;
        f32 y_offset = random_f32() - 0.5f;
        current_ray_direction = vec3_sub(current_pixel_center, camera_pos);
        current_ray_direction.x += x_offset * pixel_dist_x;
        current_ray_direction.y += y_offset * pixel_dist_y;

        vec3_inplace_add(&pixels_colors[curr_idx],
            ray_color(camera_pos, current_ray_direction, max_bounces, world));
      }
      vec3_inplace_scale(pixel_samples_scale, &pixels_colors[curr_idx]);
      current_pixel_center.x += pixel_dist_x;
		}
    current_pixel_center.y -= pixel_dist_y;
    current_pixel_center.x = p00x;
	}
  f64 time_elapsed = timer_stop_ms(time_start);
  printf("Rendering time: %fms\n", time_elapsed);
  f64 time_elapsed_s = timer_stop(time_start_s);
  printf("Rendering time: %fs\n", time_elapsed_s);

  pixels_to_ppm(pixels_colors, img_width, img_height);
}
