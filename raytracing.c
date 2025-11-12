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

typedef enum {
  MATERIAL_METAL,
  MATERIAL_LAMBERTIAN,
  MATERIAL_DIELECTRIC
} MaterialType;

typedef struct Material {
  MaterialType type;
  union {
    vec3 lambertian_albedo;
    struct {
      vec3 albedo;
      f32 fuzz;
    } metal;
    struct {
      f32 refraction_index;
    } dielectric;
  };
} Material;

Material make_lambertian(vec3 albedo) {
  Material result = {0};
  result.type = MATERIAL_LAMBERTIAN;
  result.lambertian_albedo = albedo;
  return result;
}

Material make_metal(vec3 albedo, f32 fuzz) {
  Material result = {0};
  result.type = MATERIAL_METAL;
  result.metal.albedo = albedo;
  result.metal.fuzz = fuzz;
  return result;
}

Material make_dielectric(f32 refraction_index) {
  Material result = {0};
  result.type = MATERIAL_DIELECTRIC;
  result.dielectric.refraction_index = refraction_index;
  return result;
}

typedef struct {
  vec3 point_hit;
  vec3 normal;
  float t;
  b32 is_front_face;
  Material mat;
} HitRecord;

// outward_normal is of unit length
void hit_record_set_face_normal(HitRecord *record, vec3 ray_direction, vec3 outward_normal) {
    record->is_front_face = vec3_dot_prod(ray_direction, outward_normal) < 0;
    record->normal = record->is_front_face ? outward_normal : vec3_sign_flip(outward_normal);
}

typedef enum {
  VIS_OBJECT_SPHERE
} VisObject;

typedef struct {
  vec3 center;
  f32 radius;
  Material mat;
} Sphere;

Sphere make_sphere(vec3 center, f32 radius, Material mat) {
  Sphere s = { center, radius, mat };
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

b32 hit(Hittable *hittable, float ray_tmin, float ray_tmax, vec3 ray_origin,
    vec3 ray_direction, HitRecord *record) {
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
      record->mat = hittable->sphere.mat;
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

vec3 refract(vec3 uv, vec3 n, f32 etai_over_etat) {
  f32 cos_theta = fmin(vec3_dot_prod(vec3_sign_flip(uv), n), 1.0f);
  vec3 r_out_perpendicular = vec3_scale(etai_over_etat, vec3_add(uv, vec3_scale(cos_theta, n)));
  f32 r_out_perp_len = vec3_length(r_out_perpendicular);
  vec3 r_out_parallel = vec3_scale(-sqrtf(fabsf(1.0f - r_out_perp_len * r_out_perp_len)), n);
  return vec3_add(r_out_perpendicular, r_out_parallel);
}

vec3 reflect(vec3 direction, vec3 normal) {
  vec3 reflected = vec3_sub(direction,
      vec3_scale(2.0f * vec3_dot_prod(direction, normal), normal));
  return reflected;
}

f32 reflectance(f32 cos, f32 refraction_index) {
  /* Use Shlick's approximation */
  f32 r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * powf(1.0f - cos, 5.0f);
}

typedef struct {
  b32 is_hit;
  vec3 direction;
  vec3 attenuation;
} ScatterRes;

ScatterRes material_scatter(Material *mat, vec3 r_in_direction, vec3 normal, b32 is_front_face) {
  ScatterRes result = {0};
  switch (mat->type) {
    case MATERIAL_LAMBERTIAN: {
      result.is_hit = 1;
      result.direction = vec3_add(normal, vec3_random_unit_vector());
      if (vec3_is_near_zero(result.direction))
        result.direction = normal;
      result.attenuation = mat->lambertian_albedo;
    } break;
    case MATERIAL_METAL: {
      vec3 reflected = reflect(r_in_direction, normal);
      result.direction = vec3_add(vec3_to_unit_vec(reflected),
          vec3_scale(mat->metal.fuzz, vec3_random_unit_vector()));
      result.attenuation = mat->metal.albedo;
      result.is_hit = vec3_dot_prod(result.direction, normal) > 0;
    } break;
    case MATERIAL_DIELECTRIC: {
      result.attenuation = Color(1.0f, 1.0f, 1.0f);
      f32 ri =
        is_front_face ? (1.0f / mat->dielectric.refraction_index) : mat->dielectric.refraction_index;
      vec3 unit_direction = vec3_to_unit_vec(r_in_direction);
      f32 cos_theta = fmin(vec3_dot_prod(vec3_sign_flip(unit_direction), normal), 1.0f);
      f32 sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

      b32 is_unable_to_refract = ri * sin_theta > 1.0f;
      if (is_unable_to_refract || reflectance(cos_theta, ri) > random_f32())
        result.direction = reflect(unit_direction, normal);
      else
        result.direction = refract(unit_direction, normal, ri);
      result.is_hit = 1;
    } break;
  }
  return result;
}

vec3 ray_color(vec3 origin, vec3 direction, i32 max_bounces, HittableList world) {
  if (max_bounces <= 0)
    return Color(0, 0, 0);

  HitRecord record = {0};
  if (hittable_list_hit(world, origin, direction, 0.001f, INFINITY, &record)) {
    ScatterRes scatter = material_scatter(&record.mat, direction, record.normal, record.is_front_face);
    if (scatter.is_hit) {
      return vec3_comp_scale(scatter.attenuation,
          ray_color(record.point_hit, scatter.direction, max_bounces - 1, world));
    } else {
      return Color(0, 0, 0);
    }
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

vec3 random_in_unit_disk() {
  while (1) {
    vec3 rand = Vec3(random_f32_bound(-1.0f, 1.0f), random_f32_bound(-1.0f, 1.0f), 0.0f);
    f32 rand_len = vec3_length(rand);
    if (rand_len * rand_len < 1.0f)
      return rand;
  }
}

vec3 defocus_disk_sample(vec3 center, vec3 defocus_disk_u, vec3 defocus_disk_v) {
  vec3 p = random_in_unit_disk();
  return vec3_add(
      center, vec3_add(vec3_scale(p.x, defocus_disk_u), vec3_scale(p.y, defocus_disk_v)));
}

i32 main() {
  vec3_to_unit_vec_test();

  HittableList world = {0};
  Material default_mat = make_lambertian(Vec3(0.5f, 0.5f, 0.5f));
  Hittable hittables[500];

  world.objects = hittables;
  world.count = 0;

  Material material_ground = make_lambertian(Color(0.5f, 0.5f, 0.5f));
  Hittable ground = { VIS_OBJECT_SPHERE, make_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, material_ground) };
  hittables[world.count++] = ground;

  for (i32 a = -11; a < 11; ++a) {
    for (i32 b = -11; b < 11; ++b) {
      f32 material_choice = random_f32();
      vec3 center = { a + 0.9f*random_f32(), 0.2f, b + 0.9f*random_f32() };

      if (vec3_length(vec3_sub(center, Vec3(4.0f, 0.2f, 0.0f))) > 0.9f) {
        Material current_material;
        if (material_choice < 0.8f) {
          vec3 albedo = vec3_comp_scale(vec3_random(), vec3_random());
          current_material = make_lambertian(albedo);
        } else if (material_choice < 0.95f) {
          vec3 albedo = vec3_random_bound(0.5f, 1.0f);
          f32 fuzz = random_f32_bound(0.0f, 0.5f);
          current_material = make_metal(albedo, fuzz);
        } else {
          current_material = make_dielectric(1.5f);
        }
        Hittable obj = { VIS_OBJECT_SPHERE, make_sphere(center, 0.2f, current_material) };
        hittables[world.count++] = obj;
      }
    }
  }

  Material mat1 = make_dielectric(1.5f);
  Hittable big_sphere1 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, mat1) };
  hittables[world.count++] = big_sphere1;

  Material mat2 = make_lambertian(Vec3(0.4f, 0.2f, 0.1f));
  Hittable big_sphere2 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, mat2) };
  hittables[world.count++] = big_sphere2;

  Material mat3 = make_metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f);
  Hittable big_sphere3 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, mat3) };
  hittables[world.count++] = big_sphere3;

  i32 samples_per_pixel = 10;
  f32 pixel_samples_scale = 1.0f / (f32)samples_per_pixel;
  i32 max_bounces = 50;

  f32 vfov = 20.0f;
  vec3 lookfrom = Vec3(13.0f, 2.0f, 3.0f);
  vec3 lookat = Vec3(0.0f, 0.0f, 0.0f);
  vec3 view_up = Vec3(0.0f, 1.0f, 0.0f);

	i32 img_width = 1200;
	f32 img_ratio = 16.0f/9.0f;
	i32 img_height = (int)(img_width / img_ratio);
	
	/* f32 camera_viewport_dist = vec3_length(vec3_sub(lookfrom, lookat)); */
  f32 defocus_angle = 0.6f;
  f32 focus_dist = 10.0f;
  f32 theta = vfov * (pi / 180.0f);
  f32 h = tanf(theta / 2.0f);
	f32 viewport_height = 2.0f * h * focus_dist;
	f32 viewport_ratio = (f32)img_width /  (f32)img_height;
	f32 viewport_width = viewport_height * viewport_ratio;
	vec3 camera_pos = lookfrom;

  vec3 cam_w = vec3_to_unit_vec(vec3_sub(lookfrom, lookat));
  vec3 cam_u = vec3_to_unit_vec(vec3_cross_product(view_up, cam_w));
  vec3 cam_v = vec3_cross_product(cam_w, cam_u);
	
  vec3 viewport_v = vec3_scale(viewport_height, vec3_sign_flip(cam_v));
  vec3 viewport_u = vec3_scale(viewport_width, cam_u);

  vec3 pixel_delta_v = vec3_scale(1.0f/(f32)img_height, viewport_v);
  vec3 pixel_delta_u = vec3_scale(1.0f/(f32)img_width, viewport_u);
	
	vec3 viewport_left_upper_corner = camera_pos;
  vec3_inplace_sub(&viewport_left_upper_corner, vec3_scale(focus_dist, cam_w));
  vec3_inplace_sub(&viewport_left_upper_corner, vec3_scale(0.5f, viewport_u));
  vec3_inplace_sub(&viewport_left_upper_corner, vec3_scale(0.5f, viewport_v));

  f32 defocus_radius = focus_dist * tan((defocus_angle / 2.0f) * pi / 180.0f);
  vec3 defocus_disk_u = vec3_scale(defocus_radius, cam_u);
  vec3 defocus_disk_v = vec3_scale(defocus_radius, cam_v);

	vec3 current_pixel_center = viewport_left_upper_corner;
  vec3_inplace_add(&current_pixel_center, vec3_scale(0.5f, pixel_delta_v));
  vec3_inplace_add(&current_pixel_center, vec3_scale(0.5f, pixel_delta_u));
	
  vec3 *pixels_colors = calloc(img_height * img_width, sizeof(vec3));

  f64 time_start = timer_start_ms();
  f64 time_start_s = timer_start();
	for (i32 i = 0; i < img_height; ++i) {
		for (i32 j = 0; j < img_width; ++j) {
      u64 curr_idx = i * img_width + j;
      vec3 current_ray_direction = 
          vec3_add(
            current_pixel_center, vec3_add(
              vec3_scale((f32)i, pixel_delta_v),
              vec3_scale((f32)j, pixel_delta_u)));
      for (i32 sample_num = 0; sample_num < samples_per_pixel; ++sample_num) {
        vec3 ray_origin = (defocus_angle <= 0) ?
          camera_pos : defocus_disk_sample(camera_pos, defocus_disk_u, defocus_disk_v);
        f32 x_offset = random_f32() - 0.5f;
        f32 y_offset = random_f32() - 0.5f;
        vec3 sampling_ray_direction = vec3_sub(current_ray_direction, ray_origin);
        vec3_inplace_add(&sampling_ray_direction, vec3_scale(x_offset, pixel_delta_u));
        vec3_inplace_add(&sampling_ray_direction, vec3_scale(y_offset, pixel_delta_v));

        vec3_inplace_add(&pixels_colors[curr_idx],
            ray_color(ray_origin, sampling_ray_direction, max_bounces, world));
      }
      vec3_inplace_scale(pixel_samples_scale, &pixels_colors[curr_idx]);
		}
	}
  f64 time_elapsed = timer_stop_ms(time_start);
  printf("Rendering time: %fms\n", time_elapsed);
  f64 time_elapsed_s = timer_stop(time_start_s);
  printf("Rendering time: %fs\n", time_elapsed_s);

  pixels_to_ppm(pixels_colors, img_width, img_height);
}
