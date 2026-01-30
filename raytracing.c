#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#include "vec3.h"
#include "utility.h"
#include "perlin_noise.h"

#include "vec3.c" // unity build
#include "utility.c"
#include "perlin_noise.c"

#define WORLD_SIZE 500

const float pi = 3.1415926535897932385f;

typedef struct {
  u8 *data;
  u64 current_pos;
  u64 size;
} MemArena;

void mem_arena_alloc(MemArena *arena, u64 size) {
  arena->size = size;
  arena->data = malloc(size);
  arena->current_pos = 0;
}

void *mem_arena_push(MemArena *arena, u64 size) {
  u8 *result = NULL;
  if (arena->current_pos + size < arena->size) {
    result = arena->data + arena->current_pos;
    arena->current_pos += size;
  }
  return (void *)result;
}

typedef struct {
  vec3 albedo;
} SolidColor;

typedef struct Texture Texture;

typedef struct {
  f32 inverted_scale;
  Texture *even;
  Texture *odd;
} Checker;

typedef struct {
  vec3 *pixels;
  i32 width;
  i32 height;
} ImageTexture;

typedef struct {
  PerlinNoise *noise;
  f32 scale;
} NoiseTexture;

typedef u32 TEXTURE;
enum {
  TEXTURE_SOLID_COLOR,
  TEXTURE_CHECKER,
  TEXTURE_IMAGE,
  TEXTURE_NOISE
};

struct Texture {
  TEXTURE type;
  union {
    SolidColor solid_color;
    Checker checker;
    ImageTexture image_texture;
    NoiseTexture noise_texture;
  };
};

typedef struct {
  f32 x0, x1;
  f32 y0, y1;
  f32 z0, z1;
} Aabb;

typedef struct {
  vec3 center;
  vec3 direction;
} MovementPath;

typedef struct {
  vec3 origin;
  vec3 direction;
  f32 intersection_time;
} Ray;

typedef u32 MaterialType;
enum {
  MATERIAL_LAMBERTIAN,
  MATERIAL_METAL,
  MATERIAL_DIELECTRIC
};

typedef struct Material {
  MaterialType type;
  union {
    Texture *lambertian_texture;
    struct {
      vec3 albedo;
      f32 fuzz;
    } metal;
    struct {
      f32 refraction_index;
    } dielectric;
  };
} Material;

typedef struct {
  b32 is_hit;
  vec3 direction;
  vec3 attenuation;
} ScatterRes;

typedef struct {
  MovementPath movement_path;
  f32 radius;
  Material mat;
  Aabb bounding_box;
} Sphere;

typedef struct Hittable Hittable;

typedef struct {
  Hittable *left;
  Hittable *right;
  Aabb bounding_box;
} BvhNode;

typedef u32 VisObject;
enum {
  VIS_OBJECT_SPHERE,
  VIS_OBJECT_BVH_NODE,
  VIS_OBJECT_HITTABLE_LIST
};

typedef struct {
  Hittable *objects;
  Aabb bounding_box;
  u64 size;
  u64 count;
} HittableList;

struct Hittable {
  VisObject type;
  union {
    Sphere sphere;
    BvhNode bvh_node;
    HittableList hlist;
  };
};

typedef u32 HITTABLE_LIST_COMP_TYPE;
enum {
  HITTABLE_LIST_COMP_TYPE_X_AXIS,
  HITTABLE_LIST_COMP_TYPE_Y_AXIS,
  HITTABLE_LIST_COMP_TYPE_Z_AXIS,
  HITTABLE_LIST_COMP_TYPE_COUNT
};

typedef struct {
  vec3 point_hit;
  vec3 normal;
  float t;
  b32 is_front_face;
  Material mat;
  f32 u;
  f32 v;
} HitRecord;

Texture make_noise_texture(PerlinNoise *noise, f32 scale) {
  Texture result = {0};
  result.type = TEXTURE_NOISE;
  result.noise_texture.noise = noise;
  result.noise_texture.scale = scale;
  return result;
}

SolidColor make_solid_color(vec3 albedo) {
  SolidColor result = { albedo };
  return result;
}

Texture make_checker_texture(f32 scale, Texture *even, Texture *odd) {
  Texture result = {0};
  result.type = TEXTURE_CHECKER;
  result.checker.inverted_scale = 1.0f / scale;
  result.checker.even = even;
  result.checker.odd = odd;
  return result;
}

Texture make_image_texture(char *file_name) {
  ImageTexture img_tex = {0};
  i32 bytes_per_pixel = 3;
  i32 dummy = 0;
  f32 *data = stbi_loadf(file_name, &img_tex.width, &img_tex.height, &dummy, bytes_per_pixel);
  img_tex.pixels = (vec3 *)data;
  Texture result = { TEXTURE_IMAGE };
  result.image_texture = img_tex;
  return result;
}

vec3 image_texture_get_pixel(ImageTexture *img_tex, i32 x, i32 y) {
  vec3 result = {0};
  if (img_tex->pixels == NULL) {
    result = Vec3(1, 0, 1);
  } else {
    x = CLAMP(x, 0, img_tex->width);
    y = CLAMP(y, 0, img_tex->height);
    result = *(img_tex->pixels + y * img_tex->width + x);
  }
  return result;
}

vec3 texture_get_color(Texture *texture, f32 u, f32 v, vec3 point) {
  vec3 result = {0};
  switch(texture->type) {
    case TEXTURE_SOLID_COLOR: {
      result = texture->solid_color.albedo;
    } break;
    case TEXTURE_CHECKER: {
      Checker *checker = &texture->checker;
      u32 x = (u32)floorf(checker->inverted_scale * point.x);
      u32 y = (u32)floorf(checker->inverted_scale * point.y);
      u32 z = (u32)floorf(checker->inverted_scale * point.z);
      
      b32 is_even = (x + y + z) % 2 == 0;

      result = is_even ?
        texture_get_color(checker->even, u, v, point) :
        texture_get_color(checker->odd, u, v, point);
    } break;
    case TEXTURE_IMAGE: {
      ImageTexture *img_tex = &texture->image_texture;
      if (img_tex->height <= 0)
        result = Vec3(0, 1, 1);
      u = CLAMP(u, 0, 1);
      v = 1.0f - CLAMP(v, 0, 1);
      result =
        image_texture_get_pixel(img_tex, (i32)(u * img_tex->width), (i32)(v * img_tex->height));
    } break;
    case TEXTURE_NOISE: {
      NoiseTexture *noise_tex = &texture->noise_texture;
      result =
        vec3_scale(1 + sinf(
            noise_tex->scale * point.z + 10 * perlin_turbulence(
                                                noise_tex->noise, 
                                                point,
                                                7)),
            Color(0.5f, 0.5f, 0.5f));
    } break;
  }
  return result;
}

MovementPath make_movement_path(vec3 center, vec3 direction) {
  MovementPath result = { center, direction };
  return result;
}

vec3 movement_path_at(MovementPath *mp, f32 t) {
  return line_at(mp->center, t, mp->direction);
}

vec3 ray_at(Ray *ray, f32 t) {
  return line_at(ray->origin, t, ray->direction);
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

ScatterRes material_scatter(Material *mat, vec3 r_in_direction, HitRecord *record) {
  ScatterRes result = {0};
  switch (mat->type) {
    case MATERIAL_LAMBERTIAN: {
      result.is_hit = 1;
      result.direction = vec3_add(record->normal, vec3_random_unit_vector());
      if (vec3_is_near_zero(result.direction))
        result.direction = record->normal;
      result.attenuation = texture_get_color(mat->lambertian_texture, 
          record->u, record->v, record->point_hit);
    } break;
    case MATERIAL_METAL: {
      vec3 reflected = reflect(r_in_direction, record->normal);
      result.direction = vec3_add(vec3_to_unit_vec(reflected),
          vec3_scale(mat->metal.fuzz, vec3_random_unit_vector()));
      result.attenuation = mat->metal.albedo;
      result.is_hit = vec3_dot_prod(result.direction, record->normal) > 0;
    } break;
    case MATERIAL_DIELECTRIC: {
      result.attenuation = Color(1.0f, 1.0f, 1.0f);
      f32 ri =
        record->is_front_face ? (1.0f / mat->dielectric.refraction_index) : mat->dielectric.refraction_index;
      vec3 unit_direction = vec3_to_unit_vec(r_in_direction);
      f32 cos_theta = fmin(vec3_dot_prod(vec3_sign_flip(unit_direction), record->normal), 1.0f);
      f32 sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

      b32 is_unable_to_refract = ri * sin_theta > 1.0f;
      if (is_unable_to_refract || reflectance(cos_theta, ri) > random_f32())
        result.direction = reflect(unit_direction, record->normal);
      else
        result.direction = refract(unit_direction, record->normal, ri);
      result.is_hit = 1;
    } break;
  }
  return result;
}

Material make_lambertian_solid_color(MemArena *arena, vec3 albedo) {
  Material result = {0};
  result.type = MATERIAL_LAMBERTIAN;
  SolidColor sc = { albedo };
  result.lambertian_texture = mem_arena_push(arena, sizeof(Texture));
  result.lambertian_texture->type = TEXTURE_SOLID_COLOR;
  result.lambertian_texture->solid_color = sc;
  return result;
}

Material make_lambertian(Texture *tex) {
  Material result = {0};
  result.type = MATERIAL_LAMBERTIAN;
  result.lambertian_texture = tex;
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

Aabb make_aabb(vec3 p1, vec3 p2) {
  Aabb result = {0};
  result.x0 = MIN(p1.x, p2.x);
  result.x1 = MAX(p1.x, p2.x);
  result.y0 = MIN(p1.y, p2.y);
  result.y1 = MAX(p1.y, p2.y);
  result.z0 = MIN(p1.z, p2.z);
  result.z1 = MAX(p1.z, p2.z);
  return result;
}

Aabb aabb_merge(Aabb *a, Aabb *b) {
  Aabb result = {0};
  result.x0 = MIN(a->x0, b->x0);
  result.x1 = MAX(a->x1, b->x1);
  result.y0 = MIN(a->y0, b->y0);
  result.y1 = MAX(a->y1, b->y1);
  result.z0 = MIN(a->z0, b->z0);
  result.z1 = MAX(a->z1, b->z1);
  return result;
}

void sphere_get_uv(vec3 *point, f32 *u, f32 *v) {
  f32 theta = acos(-point->y);
  f32 phi = atan2(-point->z, point->x) + pi;

  *u = phi / (2 * pi);
  *v = theta / pi;
}

Sphere make_sphere(vec3 center, f32 radius, Material mat) {
  Sphere s = { make_movement_path(center, Vec3(0.0f, 0.0f, 0.0f)), radius, mat };
  vec3 radius_vec = { radius, radius, radius };
  s.bounding_box = make_aabb(vec3_sub(center, radius_vec), vec3_add(center, radius_vec));
  return s;
}

Hittable make_hittable_sphere(vec3 center, f32 radius, Material mat) {
  Hittable result = { VIS_OBJECT_SPHERE, make_sphere(center, radius, mat) };
  return result;
}

Sphere make_moving_sphere(vec3 center, vec3 destination, f32 radius, Material mat) {
  Sphere s = { make_movement_path(center, destination), radius, mat };
  vec3 radius_vec = { radius, radius, radius };
  vec3 min_pos = movement_path_at(&s.movement_path, 0.0f);
  vec3 max_pos = movement_path_at(&s.movement_path, 1.0f);
  Aabb bbox1 = make_aabb(vec3_sub(min_pos, radius_vec), vec3_add(min_pos, radius_vec));
  Aabb bbox2 = make_aabb(vec3_sub(max_pos, radius_vec), vec3_add(max_pos, radius_vec));
  s.bounding_box = aabb_merge(&bbox1, &bbox2);
  return s;
}

Aabb *hittable_get_bounding_box(Hittable *hittable) {
  Aabb *result = NULL;
  switch (hittable->type) {
    case VIS_OBJECT_SPHERE: {
      result = &hittable->sphere.bounding_box;
    } break;
    case VIS_OBJECT_BVH_NODE: {
      result = &hittable->bvh_node.bounding_box;
    } break;
    case VIS_OBJECT_HITTABLE_LIST: {
      result = &hittable->hlist.bounding_box;
    } break;
  }
  return result;
}

b32 hittable_list_comparison(Hittable *arr, u64 i, u64 j, HITTABLE_LIST_COMP_TYPE type) {
  b32 comp = 0;
  Aabb *bbox1 = hittable_get_bounding_box(&arr[i]);
  Aabb *bbox2 = hittable_get_bounding_box(&arr[j]);
  switch (type) {
    case HITTABLE_LIST_COMP_TYPE_X_AXIS:
      comp = bbox1->x0 < bbox2->x0;
      break;
    case HITTABLE_LIST_COMP_TYPE_Y_AXIS:
      comp = bbox1->y0 < bbox2->y0;
      break;
    case HITTABLE_LIST_COMP_TYPE_Z_AXIS:
      comp = bbox1->z0 < bbox2->z0;
      break;
  }
  return comp;
}

void hittable_list_merge(Hittable *arr, Hittable *tmp, u64 start, u64 mid, u64 end,
    HITTABLE_LIST_COMP_TYPE type) {
  u64 start_max = mid - 1;
  u64 tmp_pos = start;
  u64 count = end - start + 1;

  while ((end <= start_max) && (mid <= end)) {
    if (hittable_list_comparison(arr, start, mid, type)) {
      tmp[tmp_pos] = arr[start];
      tmp_pos = tmp_pos + 1;
      start = start + 1;
    } else {
      tmp[tmp_pos] = arr[mid];
      tmp_pos = tmp_pos + 1;
      mid = mid + 1;
    }
  }

  while (start <= start_max) {
    tmp[tmp_pos] = arr[start];
    start = start + 1;
    tmp_pos = tmp_pos + 1;
  }

  while (mid <= end) {
    tmp[tmp_pos] = arr[mid];
    mid = mid + 1;
    tmp_pos = tmp_pos + 1;
  }

  for (u64 i = 0; i < count; ++i) {
    arr[end] = tmp[end];
    end = end - 1;
  }
}

void hittable_list_split_merge(Hittable *arr, Hittable *tmp, u64 start, u64 end,
    HITTABLE_LIST_COMP_TYPE type) {
  if (end > start) {
    u64 mid = (end + start) / 2;
    hittable_list_split_merge(arr, tmp, start, mid, type);
    hittable_list_split_merge(arr, tmp, mid + 1, end, type);
    hittable_list_merge(arr, tmp, start, mid + 1, end, type);
  }
}

Hittable hittables_copy[WORLD_SIZE] = {0};

void hittable_list_sort_bound(HittableList *list, u64 start, u64 end, 
    HITTABLE_LIST_COMP_TYPE type) {
  memcpy(hittables_copy, list->objects, WORLD_SIZE);
  hittable_list_split_merge(list->objects, hittables_copy, start, end, type);
}

Hittable *make_bvh_node_from_hittable_list_bound(MemArena *arena, HittableList *list,
    u64 start, u64 end) {
  Hittable *result = mem_arena_push(arena, sizeof(Hittable));
  result->type = VIS_OBJECT_BVH_NODE;
  u64 span = end - start;
  BvhNode *node = &result->bvh_node;

  if (span == 1) {
    node->left = node->right = mem_arena_push(arena, sizeof(Hittable));
    memcpy(node->left, &list->objects[start], sizeof(Hittable));
  } else if (span == 2) {
    node->left = mem_arena_push(arena, sizeof(Hittable));
    node->right = mem_arena_push(arena, sizeof(Hittable));
    memcpy(node->left, &list->objects[start], sizeof(Hittable));
    memcpy(node->right, &list->objects[start + 1], sizeof(Hittable));
  } else {
    HITTABLE_LIST_COMP_TYPE comp_type = random_i32_bound(0, 2);
    hittable_list_sort_bound(list, start, end, comp_type);
#if 0
    for (u64 i = start; i < end - 1; ++i) {
      b32 comp = 0;
      Aabb *debug_bbox1 = hittable_get_bounding_box(&list->objects[i]);
      Aabb *debug_bbox2 = hittable_get_bounding_box(&list->objects[i + 1]);
      switch (comp_type) {
        case HITTABLE_LIST_COMP_TYPE_X_AXIS:
          comp = debug_bbox1->x0 <= debug_bbox2->x0;
        break;
        case HITTABLE_LIST_COMP_TYPE_Y_AXIS:
          comp = debug_bbox1->y0 <= debug_bbox2->y0;
        break;
        case HITTABLE_LIST_COMP_TYPE_Z_AXIS:
          comp = debug_bbox1->z0 <= debug_bbox2->z0;
        break;
      }
      if (!comp) {
        *((int *)0) = 0;
      }
    }
#endif
    u64 mid = start + span / 2;
    node->left = make_bvh_node_from_hittable_list_bound(arena, list, start, mid);
    node->right = make_bvh_node_from_hittable_list_bound(arena, list, mid, end);
  }

  Aabb *left_bbox = hittable_get_bounding_box(node->left);
  Aabb *right_bbox = hittable_get_bounding_box(node->right);
  node->bounding_box = aabb_merge(left_bbox, right_bbox);
  return result;
}

Hittable *make_bvh_node_from_hittable_list(MemArena *arena, HittableList *list) {
  mem_arena_alloc(arena, WORLD_SIZE * sizeof(Hittable) * 2);
  return make_bvh_node_from_hittable_list_bound(arena, list, 0, list->count);
}

void hittable_list_add(HittableList *list, Hittable *hittable) {
  if (list->count < list->size) {
    list->objects[list->count++] = *hittable;
    Aabb *other_bbox = hittable_get_bounding_box(hittable);
    list->bounding_box = aabb_merge(&list->bounding_box, other_bbox);
  }
}

// outward_normal is of unit length
void hit_record_set_face_normal(HitRecord *record, Ray *ray, vec3 outward_normal) {
  record->is_front_face = vec3_dot_prod(ray->direction, outward_normal) < 0;
  record->normal = record->is_front_face ? outward_normal : vec3_sign_flip(outward_normal);
}

b32 aabb_hit(Aabb *bbox, Ray *ray, f32 ray_t0, f32 ray_t1) {
  f32 t0_x = (bbox->x0 - ray->origin.x) / ray->direction.x;
  f32 t1_x = (bbox->x1 - ray->origin.x) / ray->direction.x;

  interval_sort(&t0_x, &t1_x);
  if (!interval_overlap(t0_x, t1_x, ray_t0, ray_t1))
    return 0;

  f32 t0_y = (bbox->y0 - ray->origin.y) / ray->direction.y;
  f32 t1_y = (bbox->y1 - ray->origin.y) / ray->direction.y;

  interval_sort(&t0_y, &t1_y);
  if (!interval_overlap(t0_x, t1_x, t0_y, t1_y))
    return 0;

  f32 t0_z = (bbox->z0 - ray->origin.z) / ray->direction.z;
  f32 t1_z = (bbox->z1 - ray->origin.z) / ray->direction.z;

  interval_sort(&t0_z, &t1_z);
  return interval_overlap(t0_y, t1_y, t0_z, t1_z);
}

b32 hittable_list_hit(HittableList *list, Ray *ray,
    double ray_tmin, double ray_tmax, HitRecord *record);

b32 hit(Hittable *hittable, f32 ray_tmin, f32 ray_tmax, Ray *ray, HitRecord *record) {
  switch (hittable->type) {
    case VIS_OBJECT_SPHERE: {
      vec3 current_sphere_position = 
        movement_path_at(&hittable->sphere.movement_path, ray->intersection_time);
      vec3 oc = vec3_sub(current_sphere_position, ray->origin);
      float a = vec3_dot_prod(ray->direction, ray->direction);
      float b = -2.0f * vec3_dot_prod(ray->direction, oc);
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
      record->point_hit = ray_at(ray, root);
      record->mat = hittable->sphere.mat;
      vec3 outward_normal = vec3_scale(1.0f/hittable->sphere.radius,
          vec3_sub(record->point_hit, current_sphere_position));
      sphere_get_uv(&outward_normal, &record->u, &record->v);
      hit_record_set_face_normal(record, ray, outward_normal);

      return 1;
    } break;
    case VIS_OBJECT_BVH_NODE: {
      if (aabb_hit(&hittable->bvh_node.bounding_box, ray, ray_tmin, ray_tmax)) {
        b32 hit_left = hit(hittable->bvh_node.left, ray_tmin, ray_tmax, ray, record);
        b32 hit_right =
          hit(hittable->bvh_node.right, ray_tmin, hit_left ? record->t : ray_tmax, ray, record);
        return hit_left || hit_right;
      } else {
        return 0;
      }
    } break;
    case VIS_OBJECT_HITTABLE_LIST: {
      return hittable_list_hit(&hittable->hlist, ray, ray_tmin, ray_tmax, record);
    } break;
  }
  return 0;
}

vec3 ray_color(Ray *ray, i32 max_bounces, Hittable *world) {
  if (max_bounces <= 0)
    return Color(0, 0, 0);

  HitRecord record = {0};
  if (hit(world, 0.001f, INFINITY, ray, &record)) {
    ScatterRes scatter = material_scatter(&record.mat, ray->direction, &record);
    if (scatter.is_hit) {
      Ray bounce = {0};
      bounce.origin = record.point_hit;
      bounce.direction = scatter.direction;
      bounce.intersection_time = ray->intersection_time;
      return vec3_comp_scale(scatter.attenuation,
          ray_color(&bounce, max_bounces - 1, world));
    } else {
      return Color(0, 0, 0);
    }
  }

  vec3 unit_direction = vec3_to_unit_vec(ray->direction);
  f32 blend_percent = 0.5f * (unit_direction.y + 1.0f);
  return vec3_add(vec3_scale(1.0f - blend_percent, Color(1.0f, 1.0f, 1.0f)),
      vec3_scale(blend_percent, Color(0.5f, 0.7f, 1.0f)));
}

b32 hittable_list_hit(HittableList *list, Ray *ray,
    double ray_tmin, double ray_tmax, HitRecord *record) {
  HitRecord temp_record;
  b32 hit_anything = 0;
  float closest_so_far = ray_tmax;

  for (int i = 0; i < list->count; ++i) {
    if (hit(&list->objects[i], ray_tmin, closest_so_far, ray, &temp_record)) {
      hit_anything = 1;
      closest_so_far = temp_record.t;
      *record = temp_record;
    }
  }

  return hit_anything;
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

void print_bvh_info(Hittable *node) {
  Aabb *bbox = hittable_get_bounding_box(node);
  switch (node->type) {
    case VIS_OBJECT_SPHERE: {
      printf("%f-%f,%f-%f,%f-%f sphere\n", bbox->x0, bbox->x1, bbox->y0, bbox->y1, bbox->z0, bbox->z1);
    } break;
    case VIS_OBJECT_BVH_NODE: {
      printf("%f-%f,%f-%f,%f-%f bvh_node\n", bbox->x0, bbox->x1, bbox->y0, bbox->y1, bbox->z0, bbox->z1);
      print_bvh_info(node->bvh_node.left);
      print_bvh_info(node->bvh_node.right);
    } break;
  }
}

typedef struct {
  f32 img_ratio;
  i32 img_width;
  i32 samples_per_pixel;
  i32 max_bounces;
  f32 vfov;
  vec3 lookfrom;
  vec3 lookat;
  vec3 view_up;
  f32 defocus_angle;
} RenderSettings;

static RenderSettings g_default_render_settings = 
  { 16.0f/9.0f, 1200, 10, 50, 20.0f, { 13.0f, 2.0f, 3.0f },
    { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 0.6f };

void render_w_settings(HittableList *world, RenderSettings *settings) {
  MemArena nodes_arena = {0};
  Hittable *optimized_world = make_bvh_node_from_hittable_list(&nodes_arena, world);
  /*
  Hittable hworld = {0};
  hworld.type = VIS_OBJECT_HITTABLE_LIST;
  hworld.hlist = world;
  Hittable *optimized_world = &hworld;
  print_bvh_info(optimized_world);
  */

  f32 pixel_samples_scale = 1.0f / (f32)settings->samples_per_pixel;

	i32 img_width = settings->img_width;
	i32 img_height = (int)(img_width / settings->img_ratio);
	
	/* f32 camera_viewport_dist = vec3_length(vec3_sub(settings->lookfrom, settings->lookat)); */
  f32 focus_dist = 10.0f;
  f32 theta = settings->vfov * (pi / 180.0f);
  f32 h = tanf(theta / 2.0f);
	f32 viewport_height = 2.0f * h * focus_dist;
	f32 viewport_ratio = (f32)img_width /  (f32)img_height;
	f32 viewport_width = viewport_height * viewport_ratio;
	vec3 camera_pos = settings->lookfrom;

  vec3 cam_w = vec3_to_unit_vec(vec3_sub(settings->lookfrom, settings->lookat));
  vec3 cam_u = vec3_to_unit_vec(vec3_cross_product(settings->view_up, cam_w));
  vec3 cam_v = vec3_cross_product(cam_w, cam_u);
	
  vec3 viewport_v = vec3_scale(viewport_height, vec3_sign_flip(cam_v));
  vec3 viewport_u = vec3_scale(viewport_width, cam_u);

  vec3 pixel_delta_v = vec3_scale(1.0f/(f32)img_height, viewport_v);
  vec3 pixel_delta_u = vec3_scale(1.0f/(f32)img_width, viewport_u);
	
	vec3 viewport_left_upper_corner = camera_pos;
  vec3_inplace_sub(&viewport_left_upper_corner, vec3_scale(focus_dist, cam_w));
  vec3_inplace_sub(&viewport_left_upper_corner, vec3_scale(0.5f, viewport_u));
  vec3_inplace_sub(&viewport_left_upper_corner, vec3_scale(0.5f, viewport_v));

  f32 defocus_radius = focus_dist * tan((settings->defocus_angle / 2.0f) * pi / 180.0f);
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
      for (i32 sample_num = 0; sample_num < settings->samples_per_pixel; ++sample_num) {
        Ray sampling_ray = {0};
        sampling_ray.origin = (settings->defocus_angle <= 0) ?
          camera_pos : defocus_disk_sample(camera_pos, defocus_disk_u, defocus_disk_v);
        f32 x_offset = random_f32() - 0.5f;
        f32 y_offset = random_f32() - 0.5f;
        sampling_ray.direction = vec3_sub(current_ray_direction, sampling_ray.origin);
        vec3_inplace_add(&sampling_ray.direction, vec3_scale(x_offset, pixel_delta_u));
        vec3_inplace_add(&sampling_ray.direction, vec3_scale(y_offset, pixel_delta_v));
        sampling_ray.intersection_time = random_f32();

        vec3_inplace_add(&pixels_colors[curr_idx],
            ray_color(&sampling_ray, settings->max_bounces, optimized_world));
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

void render(HittableList *world) {
  render_w_settings(world, &g_default_render_settings);
}

void bouncing_spheres_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, sizeof(Texture) * WORLD_SIZE);

  Hittable hittables[WORLD_SIZE]; // remember about the merge sort copy

  HittableList world = {0};
  world.objects = hittables;
  world.size = WORLD_SIZE;

  Material default_mat = make_lambertian_solid_color(&texture_arena, Vec3(0.5f, 0.5f, 0.5f));

  Texture color_even = { TEXTURE_SOLID_COLOR, make_solid_color(Color(0.2f, 0.3f, 0.1f)) };
  Texture color_odd = { TEXTURE_SOLID_COLOR, make_solid_color(Color(0.9f, 0.9f, 0.9f)) };
  Texture checker_texture = make_checker_texture(0.32f, &color_even, &color_odd);

  Material material_ground = make_lambertian(&checker_texture);
  Hittable ground = { VIS_OBJECT_SPHERE, make_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, material_ground) };
  hittable_list_add(&world, &ground);

  for (i32 a = -11; a < 11; ++a) {
    for (i32 b = -11; b < 11; ++b) {
      f32 material_choice = random_f32();
      vec3 center = { a + 0.9f*random_f32(), 0.2f, b + 0.9f*random_f32() };

      if (vec3_length(vec3_sub(center, Vec3(4.0f, 0.2f, 0.0f))) > 0.9f) {
        Material current_material;
        if (material_choice < 0.8f) {
          vec3 albedo = vec3_comp_scale(vec3_random(), vec3_random());
          current_material = make_lambertian_solid_color(&texture_arena, albedo);
          vec3 sphere_direction = Vec3(0.0f, random_f32_bound(0.0f, 0.5f), 0.0f);
          Hittable obj = { VIS_OBJECT_SPHERE, 
            make_moving_sphere(center, sphere_direction, 0.2f, current_material) };
          hittable_list_add(&world, &obj);
        } else {
          if (material_choice < 0.95f) {
            vec3 albedo = vec3_random_bound(0.5f, 1.0f);
            f32 fuzz = random_f32_bound(0.0f, 0.5f);
            current_material = make_metal(albedo, fuzz);
          } else {
            current_material = make_dielectric(1.5f);
          }
          Hittable obj = { VIS_OBJECT_SPHERE, make_sphere(center, 0.2f, current_material) };
          hittable_list_add(&world, &obj);
        }
      }
    }
  }

  Material mat1 = make_dielectric(1.5f);
  Hittable big_sphere1 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, mat1) };
  hittable_list_add(&world, &big_sphere1);

  Material mat2 = make_lambertian_solid_color(&texture_arena, Vec3(0.4f, 0.2f, 0.1f));
  Hittable big_sphere2 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, mat2) };
  hittable_list_add(&world, &big_sphere2);

  Material mat3 = make_metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f);
  Hittable big_sphere3 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, mat3) };
  hittable_list_add(&world, &big_sphere3);

  render(&world);
}

void checkered_spheres_scene() {
  Hittable hittables[WORLD_SIZE]; // remember about the merge sort copy

  HittableList world = {0};
  world.objects = hittables;
  world.size = WORLD_SIZE;

  Texture color_even = { TEXTURE_SOLID_COLOR, make_solid_color(Color(0.2f, 0.3f, 0.1f)) };
  Texture color_odd = { TEXTURE_SOLID_COLOR, make_solid_color(Color(0.9f, 0.9f, 0.9f)) };
  Texture checker_texture = make_checker_texture(0.32f, &color_even, &color_odd);

  Hittable sphere1 = make_hittable_sphere(Vec3(0.0f, -10.0f, 0.0f), 10.0f, make_lambertian(&checker_texture));
  Hittable sphere2 = make_hittable_sphere(Vec3(0.0f, 10.0f, 0.0f), 10.0f, make_lambertian(&checker_texture));

  hittable_list_add(&world, &sphere1);
  hittable_list_add(&world, &sphere2);

  RenderSettings settings = {0};
  settings.img_ratio = 16.0f / 9.0f;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.max_bounces = 50;
  settings.vfov = 20.0f;
  settings.lookfrom = Vec3(13.0f, 2.0f, 3.0f);
  settings.lookat = Vec3(0.0f, 0.0f, 0.0f);
  settings.view_up = Vec3(0.0f, 0.1f, 0.0f);
  settings.defocus_angle = 0.0f;

  render_w_settings(&world, &settings);
}

void earth_scene() {
  Texture earth_texture = make_image_texture("earthmap.jpg");
  Material earth_surface = make_lambertian(&earth_texture);

  Hittable globe = make_hittable_sphere(Vec3(0,0,0), 2.0f, earth_surface);

  Hittable hittables[WORLD_SIZE]; // remember about the merge sort copy

  HittableList world = {0};
  world.objects = hittables;
  world.size = WORLD_SIZE;
  hittable_list_add(&world, &globe);

  RenderSettings settings = g_default_render_settings;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.lookfrom = Vec3(0, 0, 12);
  settings.lookat = Vec3(0, 0, 0);
  settings.defocus_angle = 0;

  render_w_settings(&world, &settings);
}

void perlin_spheres_scene() {
  PerlinNoise noise = make_perlin_noise();
  Texture noise_tex = make_noise_texture(&noise, 4);
  Hittable ground = make_hittable_sphere(Vec3(0, -1000, 0), 1000, make_lambertian(&noise_tex));
  Hittable sphere = make_hittable_sphere(Vec3(0, 2, 0), 2, make_lambertian(&noise_tex));

  Hittable hittables[WORLD_SIZE]; // remember about the merge sort copy

  HittableList world = {0};
  world.objects = hittables;
  world.size = WORLD_SIZE;
  hittable_list_add(&world, &ground);
  hittable_list_add(&world, &sphere);

  RenderSettings settings = g_default_render_settings;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.lookfrom = Vec3(13, 2, 3);
  settings.lookat = Vec3(0, 0, 0);
  settings.defocus_angle = 0;

  render_w_settings(&world, &settings);
}

i32 main() {
  // vec3_to_unit_vec_test();

  // bouncing_spheres_scene();
  // checkered_spheres_scene();
  // earth_scene();
  perlin_spheres_scene();
}
