#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <curand_kernel.h>
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#include "utility.h"
#include "vec3.h"
#include "perlin_noise.h"

// unity build
#include "perlin_noise.c"

#define WORLD_SIZE 1000

#define PTR_TO_INT(p) (u64)((u8*)p - (u8*)0)
#define INT_TO_PTR(n) (u8*)((u8*)0 + (n))

//NOTE: alignment must be a power of 2
__device__ __host__
inline u64 align_up(u64 p, u64 alignment) {
  return (p + alignment - 1) & ~(alignment - 1);
}

__device__ __host__
inline u8 *align_ptr_up(u8 *p, u64 alignment) {
  return INT_TO_PTR(align_up(PTR_TO_INT(p), alignment));
}

typedef struct {
  u8 *data;
  u64 current_pos;
  u64 size;
} MemArena;

void mem_arena_alloc(MemArena *arena, u64 size) {
  arena->size = size;
  arena->data = (u8 *)malloc(size);
  arena->current_pos = 0;
}

void *mem_arena_push(MemArena *arena, u64 size) {
  u8 *result = NULL;
  if (arena->current_pos + size <= arena->size) {
    result = arena->data + arena->current_pos;
    arena->current_pos += size;
  }
  //arena->current_pos = align_up(arena->current_pos, 256);
  return (void *)result;
}

void mem_arena_free(MemArena *arena) {
  free(arena->data);
  arena->current_pos = 0;
  arena->size = 0;
}

typedef MemArena GpuMemArena;

void gpu_mem_arena_alloc(MemArena *arena, u64 size) {
  arena->size = size;
  (u8 *)cudaMalloc(&arena->data, size);
  arena->current_pos = 0;
}

void *gpu_mem_arena_push(MemArena *arena, u64 size) {
  return mem_arena_push(arena, size);
}

void gpu_mem_arena_free(MemArena *arena) {
  cudaFree(arena->data);
  arena->current_pos = 0;
  arena->size = 0;
}

typedef u16 HittableRef;

typedef struct {
  HittableRef *data;
  u64 count;
  u64 size;
} HittableRefList;

typedef struct {
  vec3 albedo;
} SolidColor;

typedef struct Texture Texture;

typedef struct {
  u64 idx;
} TextureRef;

typedef struct {
  f32 inverted_scale;
  TextureRef even;
  TextureRef odd;
} Checker;

typedef struct {
  vec3 *pixels;
  u64 loaded_pixels_idx;
  i32 width;
  i32 height;
} ImageTexture;

typedef struct {
  PerlinNoise *noise;
  u64 perlin_noise_idx;
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
  MemArena arena;
  u64 count;
} TextureList;

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
  MATERIAL_DIELECTRIC,
  MATERIAL_DIFFUSE_LIGHT
};

typedef struct Material {
  MaterialType type;
  union {
    TextureRef lambertian_texture;
    struct {
      vec3 albedo;
      f32 fuzz;
    } metal;
    struct {
      f32 refraction_index;
    } dielectric;
    TextureRef diffuse_light_texture;
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

typedef struct {
  vec3 q;
  vec3 u;
  vec3 v;
  f32 d;
  vec3 w;
  vec3 normal;
} Geometry;

typedef struct {
  Geometry geom;
  Material mat;
  Aabb bounding_box;
} Quad;

typedef struct {
  Geometry geom;
  Material mat;
  Aabb bounding_box;
} Triangle;

typedef struct Hittable Hittable;
typedef HittableRef WorldRef;

typedef struct {
  WorldRef left;
  WorldRef right;
  Aabb bounding_box;
} BvhNode;

typedef u32 VisObject;
enum {
  VIS_OBJECT_SPHERE,
  VIS_OBJECT_BVH_NODE,
  VIS_OBJECT_TRIANGLE,
  VIS_OBJECT_QUAD,
  VIS_OBJECT_TRANSLATION,
  VIS_OBJECT_Y_ROTATION,
  VIS_OBJECT_HITTABLE_STRUCTURE
};

typedef struct {
  Hittable *objects;
  Aabb bounding_box;
  u64 size;
  u64 count;
} HittableList;

typedef struct {
  HittableRef idx;
  Aabb bounding_box;
  u64 size;
  u64 count;
} HittableStructure;

typedef struct {
  HittableRef idx;
  vec3 offset;
  Aabb bounding_box;
} Translation;

typedef struct {
  HittableRef idx;
  f32 sin_theta;
  f32 cos_theta;
  Aabb bounding_box;
} YRotation;

struct Hittable {
  VisObject type;
  union {
    Sphere sphere;
    BvhNode bvh_node;
    Triangle triangle;
    Quad quad;
    Translation translation;
    YRotation y_rotation;
    HittableStructure hstructure;
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

typedef struct {
  vec3 attenuation;
  vec3 emission;
} RayColorStackData;

typedef struct {
  RayColorStackData *data;
  u8 max_count;
  u8 count;
} RayColorStack;

typedef struct {
  HittableRef *data;
  u16 max_count;
  u16 count;
} HittableRefStack;

typedef struct {
  HittableList hittables;
  TextureList textures;
  HittableRefList world;
  WorldRef root_bvh_node;
} SceneData;

typedef struct {
  RayColorStack rc_stack;
  HittableRefStack hit_stack;
  curandState rng;
} PerPixelGpuData;

__host__ __device__
inline HittableRef wref_to_href(HittableRefList *world, WorldRef ref) {
  return world->data[ref];
}

HittableRefList make_hittable_ref_list(MemArena *arena, u64 size) {
  HittableRefList result = {0};
  result.data = (HittableRef *)mem_arena_push(arena, sizeof(HittableRef) * size);
  if (result.data)
    result.size = size;
  return result;
}

u64 hittable_ref_list_add(HittableRefList *list, HittableRef tex) {
  u64 result = list->count;
  if (list->count < list->size)
    list->data[list->count++] = tex;
  return result;
}

TextureRef texture_list_add(TextureList *list, Texture tex) {
  TextureRef result = {0};
  switch (tex.type) {
    case TEXTURE_SOLID_COLOR:
    case TEXTURE_CHECKER: {
      result.idx = list->arena.current_pos;
      *((Texture *)mem_arena_push(&list->arena, sizeof(Texture))) = tex;
    } break;
    case TEXTURE_IMAGE: {
      result.idx = list->arena.current_pos;
      Texture *loaded_texture = (Texture *)mem_arena_push(&list->arena, sizeof(Texture));
      *loaded_texture = tex;
      loaded_texture->image_texture.loaded_pixels_idx = list->arena.current_pos;
      u64 image_size = sizeof(vec3) * tex.image_texture.width * tex.image_texture.height;
      vec3 *pixels = (vec3 *)mem_arena_push(&list->arena, image_size);
      memcpy(pixels, tex.image_texture.pixels, image_size);
    } break;
    case TEXTURE_NOISE: {
      result.idx = list->arena.current_pos;
      Texture *loaded_texture = (Texture *)mem_arena_push(&list->arena, sizeof(Texture));
      tex.noise_texture.perlin_noise_idx = list->arena.current_pos;
      *loaded_texture = tex;
      PerlinNoise *noise = (PerlinNoise *)mem_arena_push(&list->arena, sizeof(PerlinNoise));
      memcpy(noise, tex.noise_texture.noise, sizeof(PerlinNoise));
    } break;
  }
  list->count++;
  return result;
}

TextureList make_texture_list(MemArena *arena) {
  TextureList result = {0};
  result.arena = *arena;
  return result;
}

__host__ __device__
inline Texture *texture_list_get(TextureList *list, TextureRef ref) {
  return (Texture *)(&list->arena.data[ref.idx]);
}

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

Texture make_solid_color_texture(vec3 albedo) {
  Texture result = { TEXTURE_SOLID_COLOR };
  result.solid_color.albedo = albedo;
  return result;
}

Texture make_checker_texture(f32 scale, TextureRef even, TextureRef odd) {
  Texture result = {0};
  result.type = TEXTURE_CHECKER;
  result.checker.inverted_scale = 1.0f / scale;
  result.checker.even = even;
  result.checker.odd = odd;
  return result;
}

Texture make_image_texture(const char *file_name) {
  ImageTexture img_tex = {0};
  i32 bytes_per_pixel = 3;
  i32 dummy = 0;
  f32 *data = stbi_loadf(file_name, &img_tex.width, &img_tex.height, &dummy, bytes_per_pixel);
  img_tex.pixels = (vec3 *)data;
  Texture result = { TEXTURE_IMAGE };
  result.image_texture = img_tex;
  return result;
}

__host__ __device__
inline vec3 image_texture_get_pixel(TextureList *textures, ImageTexture *img_tex, i32 x, i32 y) {
  vec3 result = {0};
  if (img_tex->pixels == NULL) {
    result = Vec3(1, 0, 1);
  } else {
    x = CLAMP(x, 0, img_tex->width);
    y = CLAMP(y, 0, img_tex->height);
    result = ((vec3 *)&textures->arena.data[img_tex->loaded_pixels_idx])[y * img_tex->width + x];
  }
  return result;
}

__host__ __device__
inline vec3 texture_get_color(TextureList *textures, TextureRef tex, f32 u, f32 v, vec3 point) {
  vec3 result = {0};
  Texture *texture = texture_list_get(textures, tex);
  u32 type = texture->type;
  switch(type) {
    case TEXTURE_SOLID_COLOR: {
      result = texture->solid_color.albedo;
    } break;
    case TEXTURE_CHECKER: {
      Checker *checker = &texture->checker;
      i32 x = (i32)floorf(checker->inverted_scale * point.x);
      i32 y = (i32)floorf(checker->inverted_scale * point.y);
      i32 z = (i32)floorf(checker->inverted_scale * point.z);
      
      b32 is_even = (x + y + z) % 2 == 0;

      result = is_even ?
        texture_get_color(textures, checker->even, u, v, point) :
        texture_get_color(textures, checker->odd, u, v, point);
    } break;
    case TEXTURE_IMAGE: {
      ImageTexture *img_tex = &texture->image_texture;
      if (img_tex->height <= 0)
        result = Vec3(0, 1, 1);
      u = CLAMP(u, 0, 1);
      v = 1.0f - CLAMP(v, 0, 1);
      result =
        image_texture_get_pixel(textures, img_tex, (i32)(u * img_tex->width), (i32)(v * img_tex->height));
    } break;
    case TEXTURE_NOISE: {
      NoiseTexture *noise_tex = &texture->noise_texture;
      result =
        vec3_scale(1 + sinf(
            noise_tex->scale * point.z + 10 * perlin_turbulence(
                                                (PerlinNoise *)&textures->arena.data[noise_tex->perlin_noise_idx], 
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

__host__ __device__
inline vec3 movement_path_at(MovementPath *mp, f32 t) {
  return line_at(mp->center, t, mp->direction);
}

__host__ __device__
inline vec3 ray_at(Ray *ray, f32 t) {
  return line_at(ray->origin, t, ray->direction);
}

__host__ __device__
inline vec3 refract(vec3 uv, vec3 n, f32 etai_over_etat) {
  f32 cos_theta = fmin(vec3_dot_prod(vec3_sign_flip(uv), n), 1.0f);
  vec3 r_out_perpendicular = vec3_scale(etai_over_etat, vec3_add(uv, vec3_scale(cos_theta, n)));
  f32 r_out_perp_len = vec3_length(r_out_perpendicular);
  vec3 r_out_parallel = vec3_scale(-sqrtf(fabsf(1.0f - r_out_perp_len * r_out_perp_len)), n);
  return vec3_add(r_out_perpendicular, r_out_parallel);
}

__host__ __device__
inline vec3 reflect(vec3 direction, vec3 normal) {
  vec3 reflected = vec3_sub(direction,
      vec3_scale(2.0f * vec3_dot_prod(direction, normal), normal));
  return reflected;
}

__host__ __device__
inline f32 reflectance(f32 cos, f32 refraction_index) {
  /* Use Shlick's approximation */
  f32 r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
  r0 = r0 * r0;
  return r0 + (1.0f - r0) * powf(1.0f - cos, 5.0f);
}

__host__ __device__
inline vec3 material_emit(TextureList *textures, Material *mat, f32 u, f32 v, vec3 point) {
  vec3 result = {0};
  switch (mat->type) {
    case MATERIAL_DIFFUSE_LIGHT: {
      result = texture_get_color(textures, mat->diffuse_light_texture, u, v, point);
    } break;
    default:
      result = Vec3(0, 0, 0);
  }
  return result;
}

// random_test (0, 1]
__host__ __device__ inline ScatterRes 
material_scatter_common(TextureList *textures, Material *mat, vec3 r_in_direction, HitRecord *record,
    vec3 random_unit_vector, f32 random_test) {
  ScatterRes result = {0};
  switch (mat->type) {
    case MATERIAL_LAMBERTIAN: {
      result.is_hit = 1;
      result.direction = vec3_add(record->normal, random_unit_vector);
      if (vec3_is_near_zero(result.direction))
        result.direction = record->normal;
      result.attenuation = texture_get_color(textures, mat->lambertian_texture, 
          record->u, record->v, record->point_hit);
    } break;
    case MATERIAL_METAL: {
      vec3 reflected = reflect(r_in_direction, record->normal);
      result.direction = vec3_add(vec3_to_unit_vec(reflected),
          vec3_scale(mat->metal.fuzz, random_unit_vector));
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
      if (is_unable_to_refract || reflectance(cos_theta, ri) > random_test)
        result.direction = reflect(unit_direction, record->normal);
      else
        result.direction = refract(unit_direction, record->normal, ri);
      result.is_hit = 1;
    } break;
  }
  return result;
}

inline ScatterRes material_scatter(TextureList *textures, Material *mat, vec3 r_in_direction, HitRecord *record) {
  return material_scatter_common(textures, mat, r_in_direction, record,
      vec3_random_unit_vector(), random_f32());
}

__device__ inline 
ScatterRes cuda_material_scatter(TextureList *textures, curandState *rng, Material *mat,
    vec3 r_in_direction, HitRecord *record) {
  return material_scatter_common(textures, mat, r_in_direction, record,
      cuda_vec3_random_unit_vector(rng), curand_uniform(rng));
}

Material make_lambertian_solid_color(MemArena *arena, vec3 albedo) {
  Material result = {0};
  return result;
}

Material make_lambertian(TextureRef tex) {
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

Material make_diffuse_light(TextureRef texture) {
  Material result = {0};
  result.type = MATERIAL_DIFFUSE_LIGHT;
  result.diffuse_light_texture = texture;
  return result;
}

Aabb aabb_add_offset(Aabb bbox, vec3 offset) {
  Aabb result = bbox;
  result.x0 += offset.x;
  result.x1 += offset.x;
  result.y0 += offset.y;
  result.y1 += offset.y;
  result.z0 += offset.z;
  result.z1 += offset.z;
  return result;
}

void aabb_pad_to_minimums(Aabb *bbox) {
  f32 delta = 0.0001f;
  if ((bbox->x1 - bbox->x0) < delta)
    interval_expand(&bbox->x0, &bbox->x1, delta);
  if ((bbox->y1 - bbox->y0) < delta)
    interval_expand(&bbox->y0, &bbox->y1, delta);
  if ((bbox->z1 - bbox->z0) < delta)
    interval_expand(&bbox->z0, &bbox->z1, delta);
}

Aabb make_aabb(vec3 p1, vec3 p2) {
  Aabb result = {0};
  result.x0 = MIN(p1.x, p2.x);
  result.x1 = MAX(p1.x, p2.x);
  result.y0 = MIN(p1.y, p2.y);
  result.y1 = MAX(p1.y, p2.y);
  result.z0 = MIN(p1.z, p2.z);
  result.z1 = MAX(p1.z, p2.z);
  aabb_pad_to_minimums(&result);
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
  aabb_pad_to_minimums(&result);
  return result;
}

__host__ __device__
inline void sphere_get_uv(vec3 *point, f32 *u, f32 *v) {
  f32 theta = acos(-point->y);
  f32 phi = atan2(-point->z, point->x) + pi;

  *u = phi / (2 * pi);
  *v = theta / pi;
}

Sphere make_sphere(vec3 center, f32 radius, Material mat) {
  Sphere result = { make_movement_path(center, Vec3(0.0f, 0.0f, 0.0f)), radius, mat };
  vec3 radius_vec = { radius, radius, radius };
  result.bounding_box = make_aabb(vec3_sub(center, radius_vec), vec3_add(center, radius_vec));
  return result;
}

Hittable make_hittable_sphere(vec3 center, f32 radius, Material mat) {
  Hittable result = { VIS_OBJECT_SPHERE, make_sphere(center, radius, mat) };
  return result;
}

Quad make_quad(vec3 q, vec3 u, vec3 v, Material mat) {
  Quad result = {0};

  Geometry geom = { q, u, v };
  vec3 n = vec3_cross_product(u, v);
  geom.w = vec3_scale(1.0f/vec3_dot_prod(n, n), n);
  geom.normal = vec3_to_unit_vec(n);
  geom.d = vec3_dot_prod(geom.normal, q);
  result.geom = geom;

  result.mat = mat;

  Aabb bbox1 = make_aabb(q, vec3_add(q, vec3_add(u, v)));
  Aabb bbox2 = make_aabb(vec3_add(q, u), vec3_add(q, v));
  result.bounding_box = aabb_merge(&bbox1, &bbox2);
  return result;
}

Triangle make_triangle(vec3 q, vec3 u, vec3 v, Material mat) {
  Triangle result = {0};

  Geometry geom = { q, u, v };
  vec3 n = vec3_cross_product(u, v);
  geom.w = vec3_scale(1.0f/vec3_dot_prod(n, n), n);
  geom.normal = vec3_to_unit_vec(n);
  geom.d = vec3_dot_prod(geom.normal, q);
  result.geom = geom;

  result.mat = mat;

  vec3 p1 = vec3_add(q, v);
  vec3 p2 = vec3_add(q, u);
  Aabb bbox = {0};
  bbox.x0 = MIN(q.x, MIN(p1.x, p2.x));
  bbox.x1 = MAX(q.x, MAX(p1.x, p2.x));
  bbox.y0 = MIN(q.y, MIN(p1.y, p2.y));
  bbox.y1 = MAX(q.y, MAX(p1.y, p2.y));
  bbox.z0 = MIN(q.z, MIN(p1.z, p2.z));
  bbox.z1 = MAX(q.z, MAX(p1.z, p2.z));
  f32 epsilon = 1e-3f;
  bbox.x0 -= epsilon;
  bbox.x1 += epsilon;
  bbox.y0 -= epsilon;
  bbox.y1 += epsilon;
  bbox.z0 -= epsilon;
  bbox.z1 += epsilon;
  result.bounding_box = bbox;
  return result;
}

Hittable make_hittable_quad(vec3 q, vec3 u, vec3 v, Material mat) {
  Hittable result = { VIS_OBJECT_QUAD };
  result.quad = make_quad(q, u, v, mat);
  return result;
}

Hittable make_hittable_triangle(vec3 q, vec3 u, vec3 v, Material mat) {
  Hittable result = { VIS_OBJECT_TRIANGLE };
  result.triangle = make_triangle(q, u, v, mat);
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
    case VIS_OBJECT_TRIANGLE: {
      result = &hittable->triangle.bounding_box;
    } break;
    case VIS_OBJECT_QUAD: {
      result = &hittable->quad.bounding_box;
    } break;
    case VIS_OBJECT_TRANSLATION: {
      result = &hittable->translation.bounding_box;
    } break;
    case VIS_OBJECT_Y_ROTATION: {
      result = &hittable->y_rotation.bounding_box;
    } break;
    case VIS_OBJECT_HITTABLE_STRUCTURE: {
      result = &hittable->hstructure.bounding_box;
    } break;
  }
  return result;
}

b32 hittable_list_comparison(HittableList *hittables, HittableRef *arr, u64 i, u64 j,
    HITTABLE_LIST_COMP_TYPE type) {
  b32 comp = 0;
  Aabb *bbox1 = hittable_get_bounding_box(&hittables->objects[arr[i]]);
  Aabb *bbox2 = hittable_get_bounding_box(&hittables->objects[arr[j]]);
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

void hittable_list_merge(HittableList *hittables, HittableRef *arr, HittableRef *tmp,
    u64 start, u64 mid, u64 end, HITTABLE_LIST_COMP_TYPE type) {
  u64 start_max = mid - 1;
  u64 tmp_pos = start;
  u64 count = end - start + 1;

  while ((start <= start_max) && (mid <= end)) {
    if (hittable_list_comparison(hittables, arr, start, mid, type)) {
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

void hittable_list_split_merge(HittableList *hittables, HittableRef *arr, HittableRef *tmp, 
    u64 start, u64 end, HITTABLE_LIST_COMP_TYPE type) {
  if (end > start) {
    u64 mid = (end + start) / 2;
    hittable_list_split_merge(hittables, arr, tmp, start, mid, type);
    hittable_list_split_merge(hittables, arr, tmp, mid + 1, end, type);
    hittable_list_merge(hittables, arr, tmp, start, mid + 1, end, type);
  }
}

void hittable_list_sort_bound(HittableList *hittables, HittableRefList *list, u64 start, u64 end, 
    HITTABLE_LIST_COMP_TYPE type) {
  MemArena copy_arena = {0};
  mem_arena_alloc(&copy_arena, list->count * sizeof(HittableRef));
  HittableRef *list_data_copy = 
    (HittableRef *)mem_arena_push(&copy_arena, list->count * sizeof(HittableRef));
  memcpy(list_data_copy, list->data, copy_arena.size);
  hittable_list_split_merge(hittables, list->data, list_data_copy, start, end - 1, type);
  mem_arena_free(&copy_arena);
}

Hittable make_hittable_translation(HittableList *hittables, HittableRef hittable_idx, vec3 offset) {
  Hittable result = { VIS_OBJECT_TRANSLATION };
  result.translation.idx = hittable_idx;
  result.translation.offset = offset;
  result.translation.bounding_box = 
    aabb_add_offset(*hittable_get_bounding_box(&hittables->objects[hittable_idx]), offset);
  return result;
}

Hittable make_hittable_y_rotation(HittableList *hittables, HittableRef hittable_idx, f32 angle) {
  Hittable result = { VIS_OBJECT_Y_ROTATION };
  result.y_rotation.idx = hittable_idx;
  f32 radians = degrees_to_radians(angle);
  result.y_rotation.sin_theta = sinf(radians);
  result.y_rotation.cos_theta = cosf(radians);
  result.y_rotation.bounding_box = *hittable_get_bounding_box(&hittables->objects[hittable_idx]);

  YRotation *yrot = &result.y_rotation;

  vec3 min = { inf_f32(), inf_f32(), inf_f32() };
  vec3 max = { neg_inf_f32(), neg_inf_f32(), neg_inf_f32() };

  for (i32 i = 0; i < 2; ++i) {
    for (i32 j = 0; j < 2; ++j) {
      for (i32 k = 0; k < 2; ++k) {
        f32 x = i*yrot->bounding_box.x1 + (1-i)*yrot->bounding_box.x0;
        f32 y = j*yrot->bounding_box.y1 + (1-j)*yrot->bounding_box.y0;
        f32 z = k*yrot->bounding_box.z1 + (1-k)*yrot->bounding_box.z0;

        f32 new_x = yrot->cos_theta*x + yrot->sin_theta*z;
        f32 new_z = -yrot->sin_theta*x + yrot->cos_theta*z;

        vec3 tmp = { new_x, y, new_z };
        min.x = MIN(min.x, tmp.x);
        max.x = MAX(max.x, tmp.x);
        min.y = MIN(min.y, tmp.y);
        max.y = MAX(max.y, tmp.y);
        min.z = MIN(min.z, tmp.z);
        max.z = MAX(max.z, tmp.z);
      }
    }
  }
  result.y_rotation.bounding_box = make_aabb(min, max);
  return result;
}

u64 hittable_list_add(HittableList *list, Hittable hittable) {
  u64 result = list->count;
  if (list->count < list->size) {
    list->objects[list->count++] = hittable;
    Aabb *other_bbox = hittable_get_bounding_box(&hittable);
    list->bounding_box = aabb_merge(&list->bounding_box, other_bbox);
  }
  return result;
}

void hittable_list_set(HittableList *list, u64 idx, Hittable hittable) {
  if (idx < list->count) {
    list->objects[idx] = hittable;
  }
}

WorldRef make_bvh_node_from_hittable_list_bound(HittableList *hittables, HittableRefList *world,
    u64 world_start, u64 world_end) {
  Hittable curr_node = {0};
  curr_node.type = VIS_OBJECT_BVH_NODE;
  BvhNode *node = &curr_node.bvh_node;

  u64 span = world_end - world_start;

  if (span == 1) {
    node->left = node->right = world_start;
  } else if (span == 2) {
    node->left = world_start;
    node->right = world_start + 1;
  } else {
    HITTABLE_LIST_COMP_TYPE comp_type = random_i32_bound(0, 2);
    // TODO: investigate, cornell box is slower with this
    hittable_list_sort_bound(hittables, world, world_start, world_end, comp_type);
#if 0
    for (u64 i = world_start; i < world_end - 1; ++i) {
      b32 comp = 0;
      Aabb *debug_bbox1 = hittable_get_bounding_box(&hittables->objects[world->data[i]]);
      Aabb *debug_bbox2 = hittable_get_bounding_box(&hittables->objects[world->data[i + 1]]);
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
    u64 mid = world_start + span / 2;
    node->left = make_bvh_node_from_hittable_list_bound(hittables, world, world_start, mid);
    node->right = make_bvh_node_from_hittable_list_bound(hittables, world, mid, world_end);
  }

  Aabb *left_bbox =
    hittable_get_bounding_box(&hittables->objects[wref_to_href(world, node->left)]);
  Aabb *right_bbox =
    hittable_get_bounding_box(&hittables->objects[wref_to_href(world, node->right)]);
  node->bounding_box = aabb_merge(left_bbox, right_bbox);
  WorldRef result = hittable_ref_list_add(world, hittable_list_add(hittables, curr_node));
  return result;
}

WorldRef make_bvh_node_from_hittable_list(HittableList *hittables, HittableRefList *world) {
  WorldRef result =
    make_bvh_node_from_hittable_list_bound(hittables, world, 0, world->count);
  return result;
}

void hittable_structure_add(HittableList *list, HittableStructure *structure, Hittable hittable) {
  if (structure->count <= structure->size) {
    hittable_list_set(list, structure->idx + structure->count, hittable);
    Aabb *other_bbox = hittable_get_bounding_box(&hittable);
    structure->bounding_box = aabb_merge(&structure->bounding_box, other_bbox);
    structure->count += 1;
  }
}

Hittable make_hittable_structure(HittableList *list, u64 size) {
  Hittable result = {0};
  result.type = VIS_OBJECT_HITTABLE_STRUCTURE;
  result.hstructure.idx = list->count;
  result.hstructure.size = size;
  for (i32 i = 0; i < size; ++i) {
    Hittable dummy = {0};
    hittable_list_add(list, dummy);
  }
  return result;
}

Hittable make_box(HittableList *world, vec3 a, vec3 b, Material mat) {
  Hittable box = make_hittable_structure(world, 6);

  vec3 min = { MIN(a.x, b.x), MIN(a.y, b.y), MIN(a.z, b.z) };
  vec3 max = { MAX(a.x, b.x), MAX(a.y, b.y), MAX(a.z, b.z) };

  vec3 dx = { max.x - min.x, 0, 0 };
  vec3 dy = { 0, max.y - min.y, 0 };
  vec3 dz = { 0, 0, max.z - min.z };

  hittable_structure_add(world, &box.hstructure, 
      make_hittable_quad(Vec3(min.x, min.y, max.z), dx, dy, mat));
  hittable_structure_add(world, &box.hstructure, 
      make_hittable_quad(Vec3(max.x, min.y, max.z), vec3_sign_flip(dz), dy, mat));
  hittable_structure_add(world, &box.hstructure, 
      make_hittable_quad(Vec3(max.x, min.y, min.z), vec3_sign_flip(dx), dy, mat));
  hittable_structure_add(world, &box.hstructure, 
      make_hittable_quad(Vec3(min.x, min.y, min.z), dz, dy, mat));
  hittable_structure_add(world, &box.hstructure, 
      make_hittable_quad(Vec3(min.x, max.y, max.z), dx, vec3_sign_flip(dz), mat));
  hittable_structure_add(world, &box.hstructure, 
      make_hittable_quad(Vec3(min.x, min.y, min.z), dx, dz, mat));
  return box;
}

Hittable make_pyramid(HittableList *world, vec3 base_a, vec3 base_b, vec3 tip, Material mat) {
  Hittable pyramid = make_hittable_structure(world, 5);

  vec3 min = { MIN(base_a.x, base_b.x), MIN(base_a.y, base_b.y), MIN(base_a.z, base_b.z) };
  vec3 max = { MAX(base_a.x, base_b.x), MAX(base_a.y, base_b.y), MAX(base_a.z, base_b.z) };

  vec3 dx = { max.x - min.x, 0, 0 };
  vec3 dy = { 0, max.y - min.y, 0 };
  vec3 dz = { 0, 0, max.z - min.z };

  vec3 p1 = Vec3(min.x, min.y, min.z);
  vec3 p2 = Vec3(max.x, min.y, min.z);
  vec3 p3 = Vec3(max.x, min.y, max.z);
  vec3 p4 = Vec3(min.x, min.y, max.z);

  hittable_structure_add(world, &pyramid.hstructure, 
      make_hittable_quad(Vec3(min.x, min.y, min.z), dx, dz, mat));
  hittable_structure_add(world, &pyramid.hstructure, 
      make_hittable_triangle(p1, dx, vec3_sub(tip, p1), mat));
  hittable_structure_add(world, &pyramid.hstructure, 
      make_hittable_triangle(p2, dz, vec3_sub(tip, p2), mat));
  hittable_structure_add(world, &pyramid.hstructure, 
      make_hittable_triangle(p3, vec3_sign_flip(dx), vec3_sub(tip, p3), mat));
  hittable_structure_add(world, &pyramid.hstructure, 
      make_hittable_triangle(p4, vec3_sign_flip(dz), vec3_sub(tip, p4), mat));
  return pyramid;
}

// outward_normal is of unit length
__host__ __device__
inline void hit_record_set_face_normal(HitRecord *record, Ray *ray, vec3 outward_normal) {
  record->is_front_face = vec3_dot_prod(ray->direction, outward_normal) < 0;
  record->normal = record->is_front_face ? outward_normal : vec3_sign_flip(outward_normal);
}

__host__ __device__
inline b32 aabb_hit(Aabb *bbox, Ray *ray, f32 ray_t0, f32 ray_t1) {
  // x axis
  f32 inv_dx = 1.0f / ray->direction.x;
  f32 t0_x = (bbox->x0 - ray->origin.x) * inv_dx;
  f32 t1_x = (bbox->x1 - ray->origin.x) * inv_dx;
  if (inv_dx < 0.0f)
    interval_sort(&t0_x, &t1_x);
  ray_t0 = MAX(ray_t0, t0_x);
  ray_t1 = MIN(ray_t1, t1_x);
  if (ray_t1 <= ray_t0)
    return 0;

  // y axis
  f32 inv_dy = 1.0f / ray->direction.y;
  f32 t0_y = (bbox->y0 - ray->origin.y) * inv_dy;
  f32 t1_y = (bbox->y1 - ray->origin.y) * inv_dy;
  if (inv_dy < 0.0f)
    interval_sort(&t0_y, &t1_y);
  ray_t0 = MAX(ray_t0, t0_y);
  ray_t1 = MIN(ray_t1, t1_y);
  if (ray_t1 <= ray_t0)
    return 0;


  // z axis
  f32 inv_dz = 1.0f / ray->direction.z;
  f32 t0_z = (bbox->z0 - ray->origin.z) * inv_dz;
  f32 t1_z = (bbox->z1 - ray->origin.z) * inv_dz;
  if (inv_dz < 0.0f)
    interval_sort(&t0_z, &t1_z);
  ray_t0 = MAX(ray_t0, t0_z);
  ray_t1 = MIN(ray_t1, t1_z);
  if (ray_t1 <= ray_t0)
    return 0;
  return 1;
}

__host__ __device__
inline b32 hittable_structure_hit(PerPixelGpuData *gd, SceneData *sd, HittableStructure *structure, Ray *ray,
    double ray_tmin, double ray_tmax, HitRecord *record);

__host__ __device__
inline b32 href_stack_is_empty(HittableRefStack *stack) {
  return stack->count == 0;
}

__host__ __device__
inline b32 href_stack_push(HittableRefStack *stack, HittableRef ref) {
  b32 result = 0;
  //assert(stack->count != stack->max_count);
  stack->data[stack->count++] = ref;
  result = 1;
  return result;
}

__host__ __device__
inline HittableRef href_stack_pop(HittableRefStack *stack) {
  HittableRef result = 0;
  if (stack->count > 0) {
    stack->count--;
    result = stack->data[stack->count];
  }
  return result;
}

__host__ __device__ inline
Hittable href_list_get(HittableRefList *refs, HittableList *hittables, HittableRef ref) {
  return hittables->objects[refs->data[ref]];
}

__host__ __device__
inline void geom_calc_hit(Geometry *geom, Ray *ray, f32 ray_tmin, f32 ray_tmax,
    f32 *out_alpha, f32 *out_beta, f32 *out_t, vec3 *out_intersection) {
  f32 denom = vec3_dot_prod(geom->normal, ray->direction);
  if (fabsf(denom) >= 1e-8) {
    *out_t = (geom->d - vec3_dot_prod(geom->normal, ray->origin)) / denom;
    if (ray_tmin <= *out_t && *out_t <= ray_tmax) {
      *out_intersection = ray_at(ray, *out_t);
      vec3 planar_hit_point = vec3_sub(*out_intersection, geom->q);
      *out_alpha = vec3_dot_prod(geom->w, vec3_cross_product(planar_hit_point, geom->v));
      *out_beta = vec3_dot_prod(geom->w, vec3_cross_product(geom->u, planar_hit_point));
    } else {
      *out_alpha = *out_beta = inf_f32();
    }
  } else {
    *out_alpha = *out_beta = inf_f32();
  }
}

__host__ __device__
inline b32 hit(PerPixelGpuData *gd, SceneData *sd, HittableRef idx,
    f32 ray_tmin, f32 ray_tmax, Ray *ray, HitRecord *record) {
  b32 result = 0;
  b32 was_last_hit = 0;
  i32 translation_cleanup_counter = 0;
  i32 yrot_cleanup_counter = 0;
  HittableRef curr_idx = 0;
  HittableRefStack *stack = &gd->hit_stack;
  href_stack_push(stack, idx);
  while (!href_stack_is_empty(stack)) {
    curr_idx = href_stack_pop(stack);
    Hittable *hittable = &sd->hittables.objects[curr_idx];
    switch (hittable->type) {
      case VIS_OBJECT_SPHERE: {
        vec3 current_sphere_position = 
          movement_path_at(&hittable->sphere.movement_path, ray->intersection_time);
        vec3 oc = vec3_sub(current_sphere_position, ray->origin);
        float a = vec3_dot_prod(ray->direction, ray->direction);
        float b = -2.0f * vec3_dot_prod(ray->direction, oc);
        float c = vec3_dot_prod(oc, oc) - hittable->sphere.radius * hittable->sphere.radius;
        float discriminant = b * b - 4*a*c;

        if (discriminant >= 0) {
          f32 sqrt_discriminant = sqrtf(discriminant);
          f32 root = (-b - sqrt_discriminant) / (2.0f*a);
          if (ray_tmin >= root || root >= ray_tmax) {
            root = (-b + sqrt_discriminant) / (2.0f*a);
          }
          if (ray_tmin < root && root < ray_tmax) {
            record->t = root;
            record->point_hit = ray_at(ray, root);
            record->mat = hittable->sphere.mat;
            vec3 outward_normal = vec3_scale(1.0f/hittable->sphere.radius,
                vec3_sub(record->point_hit, current_sphere_position));
            sphere_get_uv(&outward_normal, &record->u, &record->v);
            hit_record_set_face_normal(record, ray, outward_normal);

            ray_tmax = record->t;
            result = 1;
            if (translation_cleanup_counter || yrot_cleanup_counter)
              was_last_hit = 1;
          } 
        }
      } break;
      case VIS_OBJECT_BVH_NODE: {
        BvhNode *node = &hittable->bvh_node;
        if (aabb_hit(&node->bounding_box, ray, ray_tmin, ray_tmax)) {
          href_stack_push(stack, wref_to_href(&sd->world, node->right));
          href_stack_push(stack, wref_to_href(&sd->world, node->left));
        }
      } break;
      case VIS_OBJECT_TRIANGLE: {
        Triangle *triangle = &hittable->triangle;
        f32 alpha = 0.0f;
        f32 beta = 0.0f;
        f32 t = 0.0f;
        vec3 intersection = {0};
        geom_calc_hit(&triangle->geom, ray, ray_tmin, ray_tmax, &alpha, &beta, &t, &intersection);
        if ((0.0f <= alpha && alpha <= 1.0f) &&
            (0.0f <= beta && beta <= 1.0f) &&
            (alpha + beta <= 1.0f)) {
          record->u = alpha;
          record->v = beta;
          record->t = t;
          record->point_hit = intersection;
          record->mat = triangle->mat;
          hit_record_set_face_normal(record, ray, triangle->geom.normal);

          ray_tmax = record->t;
          result = 1;
          if (translation_cleanup_counter || yrot_cleanup_counter)
            was_last_hit = 1;
        }
      } break;
      case VIS_OBJECT_QUAD: {
        Quad *quad = &hittable->quad;
        f32 alpha = 0.0f;
        f32 beta = 0.0f;
        f32 t = 0.0f;
        vec3 intersection = {0};
        geom_calc_hit(&quad->geom, ray, ray_tmin, ray_tmax, &alpha, &beta, &t, &intersection);
        if ((0.0f <= alpha && alpha <= 1.0f) &&
            (0.0f <= beta && beta <= 1.0f)) {
          record->u = alpha;
          record->v = beta;
          record->t = t;
          record->point_hit = intersection;
          record->mat = quad->mat;
          hit_record_set_face_normal(record, ray, quad->geom.normal);

          ray_tmax = record->t;
          result = 1;
          if (translation_cleanup_counter || yrot_cleanup_counter)
            was_last_hit = 1;
        }
      } break;
      case VIS_OBJECT_TRANSLATION: {
        Translation *translation = &hittable->translation;
        if (translation_cleanup_counter == 0) {
          was_last_hit = 0;
          vec3_inplace_sub(&ray->origin, translation->offset);
          href_stack_push(stack, curr_idx);
          href_stack_push(stack, translation->idx);
          translation_cleanup_counter++;
        } else {
          vec3_inplace_add(&ray->origin, translation->offset);
          translation_cleanup_counter--;
          if (was_last_hit) {
            vec3_inplace_add(&record->point_hit, translation->offset);
          }
        }
      } break;
      case VIS_OBJECT_Y_ROTATION: {
        YRotation *y_rot = &hittable->y_rotation;
        if (yrot_cleanup_counter == 0) {
          was_last_hit = 0;
          ray->origin = Vec3(
              (y_rot->cos_theta * ray->origin.x) - (y_rot->sin_theta * ray->origin.z),
              ray->origin.y,
              (y_rot->sin_theta * ray->origin.x) + (y_rot->cos_theta * ray->origin.z)
          );

          ray->direction = Vec3(
              (y_rot->cos_theta * ray->direction.x) - (y_rot->sin_theta * ray->direction.z),
              ray->direction.y,
              (y_rot->sin_theta * ray->direction.x) + (y_rot->cos_theta * ray->direction.z)
          );
          href_stack_push(stack, curr_idx);
          href_stack_push(stack, y_rot->idx);
          yrot_cleanup_counter++;
        } else {
          ray->origin = Vec3(
              (y_rot->cos_theta * ray->origin.x) + (y_rot->sin_theta * ray->origin.z),
              ray->origin.y,
              (-y_rot->sin_theta * ray->origin.x) + (y_rot->cos_theta * ray->origin.z)
          );

          ray->direction = Vec3(
              (y_rot->cos_theta * ray->direction.x) + (y_rot->sin_theta * ray->direction.z),
              ray->direction.y,
              (-y_rot->sin_theta * ray->direction.x) + (y_rot->cos_theta * ray->direction.z)
          );
          yrot_cleanup_counter--;
          if (was_last_hit) {
            record->point_hit = Vec3(
                (y_rot->cos_theta * record->point_hit.x) + (y_rot->sin_theta * record->point_hit.z),
                record->point_hit.y,
                (-y_rot->sin_theta * record->point_hit.x) + (y_rot->cos_theta * record->point_hit.z)
            );

            record->normal = Vec3(
                (y_rot->cos_theta * record->normal.x) + (y_rot->sin_theta * record->normal.z),
                record->normal.y,
                (-y_rot->sin_theta * record->normal.x) + (y_rot->cos_theta * record->normal.z)
            );
          }
        }
      } break;
      case VIS_OBJECT_HITTABLE_STRUCTURE: {
#if 0
        result = hittable_structure_hit(gd, sd, &hittable->hstructure, ray, ray_tmin, ray_tmax, record);
#else
        HittableStructure *structure = &hittable->hstructure;
        for (HittableRef i = 0; i < structure->count; ++i) {
          href_stack_push(stack, structure->idx + i);
        }
#endif
      } break;
    }
  }
  return result;
}

__host__ __device__
inline b32 ray_color_stack_is_empty(RayColorStack *stack) {
  return stack->count == 0;
}

__host__ __device__
inline b32 ray_color_stack_push(RayColorStack *stack, vec3 attenuation, vec3 emission) {
  b32 result = 0;
  if (stack->count != stack->max_count) {
    stack->data[stack->count].attenuation = attenuation;
    stack->data[stack->count++].emission = emission;
    result = 1;
  }
  return result;
}

__host__ __device__
inline void ray_color_stack_pop(RayColorStack *stack, vec3 *attenuation, vec3 *emission) {
  if (stack->count > 0) {
    *attenuation = stack->data[stack->count - 1].attenuation;
    *emission = stack->data[stack->count - 1].emission;
    stack->count--;
  }
}

__device__
vec3 ray_color_no_bvh(SceneData *sd, PerPixelGpuData *gd, Ray *ray, vec3 background_color) {
  i32 max_bounces = gd->rc_stack.max_count;
  vec3 result = background_color;
  HitRecord record = {0};
  Ray *bounce = ray;
  b32 hit_anything = 0;
  f32 ray_tmax = inf_f32();
  while (max_bounces > 0) {
    for (WorldRef i = 0;
        i < sd->world.count && sd->hittables.objects[sd->world.data[i]].type != VIS_OBJECT_BVH_NODE; 
        ++i) {
      HitRecord temp_record = {0};
      if (hit(gd, sd, wref_to_href(&sd->world, i),
            0.001f, ray_tmax, bounce, &temp_record)) {
        ray_tmax = temp_record.t;
        hit_anything = 1;
        record = temp_record;
      }
    }
    if (hit_anything) {
      vec3 emission_color =
        material_emit(&sd->textures, &record.mat, record.u, record.v, record.point_hit);
      ScatterRes scatter =
        cuda_material_scatter(&sd->textures, &gd->rng, &record.mat, bounce->direction, &record);
      if (scatter.is_hit) {
        bounce->origin = record.point_hit;
        bounce->direction = scatter.direction;

        ray_color_stack_push(&gd->rc_stack, scatter.attenuation, emission_color);
        ray_tmax = record.t;
      } else {
        result = emission_color;
        break;
      }
      hit_anything = 0;
      ray_tmax = inf_f32();
      max_bounces--;
    } else {
      result = background_color;
      break;
    }
  }

  while (!ray_color_stack_is_empty(&gd->rc_stack)) {
    vec3 attenuation = {0};
    vec3 emission_color = {0};
    ray_color_stack_pop(&gd->rc_stack, &attenuation, &emission_color);
    vec3 scatter_color = vec3_comp_scale(attenuation, result);
    result = vec3_add(scatter_color, emission_color);
  }

  return result;
}
__device__
vec3 ray_color(SceneData *sd, PerPixelGpuData *gd, Ray *ray, vec3 background_color) {
  i32 max_bounces = gd->rc_stack.max_count;
  vec3 result = {0};
  HitRecord record = {0};
  Ray *bounce = ray;
  while (max_bounces > 0) {
    if (hit(gd, sd, wref_to_href(&sd->world, sd->root_bvh_node),
          0.001f, inf_f32(), bounce, &record)) {
      vec3 emission_color =
        material_emit(&sd->textures, &record.mat, record.u, record.v, record.point_hit);
      ScatterRes scatter =
        cuda_material_scatter(&sd->textures, &gd->rng, &record.mat, bounce->direction, &record);
      if (scatter.is_hit) {
        bounce->origin = record.point_hit;
        bounce->direction = scatter.direction;

        ray_color_stack_push(&gd->rc_stack, scatter.attenuation, emission_color);
        max_bounces--;
      } else {
        result = emission_color;
        break;
      }
    } else {
      result = background_color;
      break;
    }
  }

  while (!ray_color_stack_is_empty(&gd->rc_stack)) {
    vec3 attenuation = {0};
    vec3 emission_color = {0};
    ray_color_stack_pop(&gd->rc_stack, &attenuation, &emission_color);
    vec3 scatter_color = vec3_comp_scale(attenuation, result);
    result = vec3_add(scatter_color, emission_color);
  }

  return result;
}

__host__ __device__
inline b32 hittable_structure_hit(PerPixelGpuData *gd, SceneData *sd, HittableStructure *structure, Ray *ray,
    double ray_tmin, double ray_tmax, HitRecord *record) {
  HitRecord temp_record;
  b32 hit_anything = 0;
  float closest_so_far = ray_tmax;

  for (u32 i = structure->idx; i < structure->count; ++i) {
    if (hit(gd, sd, i, ray_tmin, closest_so_far, ray, &temp_record)) {
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

inline vec3 random_in_unit_disk() {
  while (1) {
    vec3 rand = Vec3(random_f32_bound(-1.0f, 1.0f), random_f32_bound(-1.0f, 1.0f), 0.0f);
    f32 rand_len = vec3_length(rand);
    if (rand_len * rand_len < 1.0f)
      return rand;
  }
}

inline vec3 defocus_disk_sample(vec3 center, vec3 defocus_disk_u, vec3 defocus_disk_v) {
  vec3 p = random_in_unit_disk();
  return vec3_add(
      center, vec3_add(vec3_scale(p.x, defocus_disk_u), vec3_scale(p.y, defocus_disk_v)));
}

__device__
inline vec3 cuda_random_in_unit_disk(curandState *rng) {
  while (1) {
    vec3 rand = Vec3(cuda_random_f32_bound(rng, -1.0f, 1.0f),
        cuda_random_f32_bound(rng, -1.0f, 1.0f),
        0.0f);
    f32 rand_len = vec3_length(rand);
    if (rand_len * rand_len < 1.0f)
      return rand;
  }
}

__device__ inline vec3
cuda_defocus_disk_sample(curandState *rng, vec3 center, vec3 defocus_disk_u, vec3 defocus_disk_v) {
  vec3 p = cuda_random_in_unit_disk(rng);
  return vec3_add(
      center, vec3_add(vec3_scale(p.x, defocus_disk_u), vec3_scale(p.y, defocus_disk_v)));
}

void print_bvh_info(Hittable *nodes, HittableRef *refs, WorldRef idx) {
  Hittable *node = &nodes[refs[idx]];
  Aabb *bbox = hittable_get_bounding_box(node);
  printf("===\n%f\n%f\n%f\n%f\n%f\n%f\n", bbox->x0, bbox->x1, bbox->y0, bbox->y1, bbox->z0, bbox->z1);
  switch (node->type) {
    case VIS_OBJECT_SPHERE: {
      printf("===sphere\n");
    } break;
    case VIS_OBJECT_BVH_NODE: {
      printf("===bvh_node\n");
      print_bvh_info(nodes, refs, node->bvh_node.left);
      print_bvh_info(nodes, refs, node->bvh_node.right);
    } break;
    case VIS_OBJECT_TRIANGLE: {
      printf("===triangle\n");
    } break;
    case VIS_OBJECT_QUAD: {
      printf("===quad\n");
    } break;
    case VIS_OBJECT_TRANSLATION: {
      printf("===translation\n");
    } break;
    case VIS_OBJECT_Y_ROTATION: {
      printf("===yrot\n");
    } break;
    case VIS_OBJECT_HITTABLE_STRUCTURE: {
      printf("===hstruct\n");
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
  vec3 background_color;
} RenderSettings;

static RenderSettings g_default_render_settings = 
  { 16.0f/9.0f, 1200, 100, 50, 20.0f, { 13.0f, 2.0f, 3.0f },
    { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 0.6f, {0.7f, 0.8f, 1.0f} };

#define CUDA_CHECK(x) do {                         \
    cudaError_t err = x;                           \
    if (err != cudaSuccess) {                      \
        printf("CUDA error %s at %s:%d\n",         \
               cudaGetErrorString(err),            \
               __FILE__, __LINE__);                \
        exit(1);                                   \
    }                                              \
} while (0)

#define MAX_RC_STACK 20

__global__
void cuda_calc(SceneData *sd, PerPixelGpuData *gd, vec3 *pixels_colors, i32 img_width, i32 n, 
    RenderSettings settings, vec3 pixel_delta_v, vec3 pixel_delta_u, vec3 camera_pos,
    vec3 defocus_disk_u, vec3 defocus_disk_v, f32 pixel_samples_scale,
    vec3 current_pixel_center) {
  i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
  i32 stride = blockDim.x * gridDim.x;
  if (idx >= n)
    return;

  PerPixelGpuData local_gd = gd[idx];
#if 0
  RayColorStackData local_stack_data[MAX_RC_STACK] = {0};
  local_gd.rc_stack.data = (RayColorStackData *)&local_stack_data;
  local_gd.rc_stack.max_count = MAX_RC_STACK;
#endif

  for (i32 curr_idx = idx; curr_idx < n; curr_idx += stride) {
    i32 j = curr_idx % img_width;
    i32 i = (curr_idx - j) / img_width;
    vec3 current_ray_direction = 
        vec3_add(
          current_pixel_center, vec3_add(
            vec3_scale((f32)i, pixel_delta_v),
            vec3_scale((f32)j, pixel_delta_u)));
    i32 samples_per_pixel = settings.samples_per_pixel;
    for (i32 sample_num = 0; sample_num < samples_per_pixel; ++sample_num) {
      Ray sampling_ray = {0};
      sampling_ray.origin = (settings.defocus_angle <= 0) ?
        camera_pos : cuda_defocus_disk_sample(&local_gd.rng, camera_pos, defocus_disk_u, defocus_disk_v);
      f32 x_offset = curand_uniform(&local_gd.rng) - 0.5f;
      f32 y_offset = curand_uniform(&local_gd.rng) - 0.5f;
      sampling_ray.direction = vec3_sub(current_ray_direction, sampling_ray.origin);
      vec3_inplace_add(&sampling_ray.direction, vec3_scale(x_offset, pixel_delta_u));
      vec3_inplace_add(&sampling_ray.direction, vec3_scale(y_offset, pixel_delta_v));
      sampling_ray.intersection_time = curand_uniform(&local_gd.rng);

      vec3_inplace_add(&pixels_colors[curr_idx],
          ray_color(sd,
            &local_gd,
            &sampling_ray, 
            settings.background_color));
    }
    vec3_inplace_scale(pixel_samples_scale, &pixels_colors[curr_idx]);
  }
  gd[idx].rng = local_gd.rng;
}

__global__
void init_gpu_data(PerPixelGpuData *gd, RayColorStackData *rcsd, WorldRef *hrs, 
    i32 max_bounces, u32 max_bvh_depth, u32 seed, u64 n) {
    u64 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
      gd[i].rc_stack.max_count = max_bounces;
      gd[i].rc_stack.count = 0;
      gd[i].rc_stack.data = &rcsd[i * max_bounces];

      gd[i].hit_stack.max_count = max_bvh_depth;
      gd[i].hit_stack.count = 0;
      gd[i].hit_stack.data = &hrs[i * max_bvh_depth];

      curand_init(seed, i, 0, &gd[i].rng);
    }
}

typedef struct {
  HittableList all;
  TextureList textures;
  HittableRefList bvh_include;
} World;

i32 hittable_depth(World *world, HittableRef href) {
  i32 result = 0;
  Hittable *hittable = &world->all.objects[href];
  switch (hittable->type) {
    case VIS_OBJECT_SPHERE:
    case VIS_OBJECT_TRIANGLE:
    case VIS_OBJECT_QUAD: {
      result = 1;
    } break;
    case VIS_OBJECT_TRANSLATION: {
      result = 2 + hittable_depth(world, hittable->translation.idx);
    } break;
    case VIS_OBJECT_Y_ROTATION: {
      result = 2 + hittable_depth(world, hittable->y_rotation.idx);
    } break;
    case VIS_OBJECT_HITTABLE_STRUCTURE: {
      result = hittable->hstructure.size;
    } break;
    default: {
      result = 0;
    } break;
  }
  return result;
}

i32 world_max_depth(World *world) {
  i32 result = 0;
  for (WorldRef i = 0; i < world->bvh_include.count; ++i) {
    i32 curr_depth = hittable_depth(world, wref_to_href(&world->bvh_include, i));
    if (curr_depth > result)
      result = curr_depth;
  }
  return result;
}

void render_w_settings(HittableList *hittables, TextureList *textures,
    HittableRefList *world, RenderSettings *settings) {
  WorldRef root_bvh_node = make_bvh_node_from_hittable_list(hittables, world);

  print_bvh_info(hittables->objects, world->data, root_bvh_node);

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
	
  u64 n = img_height * img_width;
  vec3 *pixels_colors;
  cudaMalloc(&pixels_colors, n * sizeof(vec3));
  cudaMemset(pixels_colors, 0, n * sizeof(vec3));

  Hittable *gpu_hittable_data;
  u64 hittable_data_size = hittables->count * sizeof(Hittable);
  cudaMalloc(&gpu_hittable_data, hittable_data_size);
  cudaMemcpy(gpu_hittable_data, hittables->objects, hittable_data_size, cudaMemcpyHostToDevice);

  HittableList gpu_hittables = *hittables;
  gpu_hittables.objects = gpu_hittable_data;

  u8 *gpu_texture_data;
  cudaMalloc(&gpu_texture_data, textures->arena.current_pos);
  cudaMemcpy(gpu_texture_data, textures->arena.data, textures->arena.current_pos, cudaMemcpyHostToDevice);

  TextureList gpu_textures = *textures;
  gpu_textures.arena.data = gpu_texture_data;

  HittableRef *gpu_hittable_refs;
  u64 hittable_refs_size = world->count * sizeof(HittableRef);
  cudaMalloc(&gpu_hittable_refs, hittable_refs_size);
  cudaMemcpy(gpu_hittable_refs, world->data, hittable_refs_size, cudaMemcpyHostToDevice);

  HittableRefList gpu_world = *world;
  gpu_world.data = gpu_hittable_refs;

  i32 block_size = 256;
  i32 num_blocks = (n + block_size - 1) / block_size;

  SceneData sd = {0};
  sd.hittables = gpu_hittables;
  sd.textures = gpu_textures;
  sd.world = gpu_world;
  sd.root_bvh_node = root_bvh_node;

  SceneData *gpu_sd;
  cudaMalloc(&gpu_sd, sizeof(SceneData));
  cudaMemcpy(gpu_sd, &sd, sizeof(SceneData), cudaMemcpyHostToDevice);

  u64 max_bvh_depth = logf(world->count) / logf(2.0f) + 1;
  World tmp = { *hittables, *textures, *world };
  // NOTE: take into account y_rotation, translation and hittable_structure hit calls
  max_bvh_depth += world_max_depth(&tmp);

  PerPixelGpuData *gd;
  cudaMalloc(&gd, n * sizeof(PerPixelGpuData));
  RayColorStackData *rc_stacks_data;
  cudaMalloc(&rc_stacks_data, n * sizeof(RayColorStackData) * settings->max_bounces);
  WorldRef *hit_stacks_data;
  cudaMalloc(&hit_stacks_data, n * sizeof(WorldRef) * max_bvh_depth);

  init_gpu_data<<<num_blocks, block_size>>>(gd, rc_stacks_data, hit_stacks_data,
      settings->max_bounces, max_bvh_depth, 1234, n);
  cudaDeviceSynchronize();

  cudaError_t err;

  f64 time_start = timer_start_ms();
  f64 time_start_s = timer_start();
  cuda_calc<<<num_blocks, block_size>>>(gpu_sd, gd, pixels_colors, img_width, n, *settings,
      pixel_delta_v, pixel_delta_u, camera_pos, defocus_disk_u, defocus_disk_v,
      pixel_samples_scale, current_pixel_center);
  //CUDA_CHECK(cudaDeviceSynchronize());
  //CUDA_CHECK(cudaGetLastError());

  err = cudaGetLastError();
  printf("cuda_calc launch error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  printf("cuda_calc sync error: %s\n", cudaGetErrorString(err));

  f64 time_elapsed = timer_stop_ms(time_start);
  printf("Rendering time: %fms\n", time_elapsed);
  f64 time_elapsed_s = timer_stop(time_start_s);
  printf("Rendering time: %fs\n", time_elapsed_s);

  vec3 *cpu_pixels_colors = (vec3 *)malloc(n * sizeof(vec3));
  cudaMemcpy(cpu_pixels_colors, pixels_colors, n * sizeof(vec3), cudaMemcpyDeviceToHost);
  pixels_to_ppm(cpu_pixels_colors, img_width, img_height);
}

void render(HittableList *hittables, TextureList *textures, HittableRefList *world) {
  render_w_settings(hittables, textures, world, &g_default_render_settings);
}

inline
WorldRef world_add(HittableList *hittables, HittableRefList *world, Hittable hittable) {
  return hittable_ref_list_add(world, hittable_list_add(hittables, hittable));
}

inline
WorldRef world_add_rotated(HittableList *hittables, HittableRefList *world, Hittable hittable, f32 angle) {
  return world_add(hittables,
      world,
      make_hittable_y_rotation(hittables, hittable_list_add(hittables, hittable), angle));
}

inline
WorldRef world_add_translated(HittableList *hittables, HittableRefList *world,
    Hittable hittable, vec3 offset) {
  return world_add(hittables,
      world,
      make_hittable_translation(hittables, hittable_list_add(hittables, hittable), offset));
}

inline
WorldRef world_add_rotated_translated(HittableList *hittables, HittableRefList *world,
    Hittable hittable, f32 angle, vec3 offset) {
  HittableRef ref = hittable_list_add(hittables,hittable);
  ref = hittable_list_add(hittables, make_hittable_y_rotation(hittables, ref, angle));
  ref = hittable_list_add(hittables,
      make_hittable_translation(hittables, ref, offset));
  return hittable_ref_list_add(world, ref);
}

void bouncing_spheres_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(5));
  TextureList textures = make_texture_list(&texture_arena);

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList world = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  Material default_mat = 
    make_lambertian(texture_list_add(&textures, make_solid_color_texture(Vec3(0.5f, 0.5f, 0.5f))));

  TextureRef color_even = texture_list_add(&textures,
      make_solid_color_texture(Vec3(0.2f, 0.3f, 0.1f)));
  TextureRef color_odd = texture_list_add(&textures,
      make_solid_color_texture(Vec3(0.9f, 0.9f, 0.9f)));
  TextureRef checker_texture = texture_list_add(&textures,
      make_checker_texture(0.32f, color_even, color_odd));

  Material material_ground = make_lambertian(checker_texture);
  Hittable ground = { VIS_OBJECT_SPHERE, make_sphere(Vec3(0.0f, -1000.0f, 0.0f), 1000.0f, material_ground) };
  hittable_ref_list_add(&world, hittable_list_add(&hittables, ground));

  for (i32 a = -11; a < 11; ++a) {
    for (i32 b = -11; b < 11; ++b) {
      f32 material_choice = random_f32();
      vec3 center = { a + 0.9f*random_f32(), 0.2f, b + 0.9f*random_f32() };

      if (vec3_length(vec3_sub(center, Vec3(4.0f, 0.2f, 0.0f))) > 0.9f) {
        Material current_material;
        if (material_choice < 0.8f) {
          vec3 albedo = vec3_comp_scale(vec3_random(), vec3_random());
          current_material = make_lambertian(texture_list_add(&textures, make_solid_color_texture(albedo)));
          vec3 sphere_direction = Vec3(0.0f, random_f32_bound(0.0f, 0.5f), 0.0f);
          Hittable obj = { VIS_OBJECT_SPHERE, 
            make_moving_sphere(center, sphere_direction, 0.2f, current_material) };
          hittable_ref_list_add(&world, hittable_list_add(&hittables, obj));
        } else {
          if (material_choice < 0.95f) {
            vec3 albedo = vec3_random_bound(0.5f, 1.0f);
            f32 fuzz = random_f32_bound(0.0f, 0.5f);
            current_material = make_metal(albedo, fuzz);
          } else {
            current_material = make_dielectric(1.5f);
          }
          Hittable obj = { VIS_OBJECT_SPHERE, make_sphere(center, 0.2f, current_material) };
          hittable_ref_list_add(&world, hittable_list_add(&hittables, obj));
        }
      }
    }
  }

  Material mat1 = make_dielectric(1.5f);
  Hittable big_sphere1 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, mat1) };
  hittable_ref_list_add(&world, hittable_list_add(&hittables, big_sphere1));

  Material mat2 = make_lambertian(texture_list_add(&textures, make_solid_color_texture(Vec3(0.4f, 0.2f, 0.1f))));
  Hittable big_sphere2 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, mat2) };
  hittable_ref_list_add(&world, hittable_list_add(&hittables, big_sphere2));

  Material mat3 = make_metal(Vec3(0.7f, 0.6f, 0.5f), 0.0f);
  Hittable big_sphere3 = { VIS_OBJECT_SPHERE, make_sphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, mat3) };
  hittable_ref_list_add(&world, hittable_list_add(&hittables, big_sphere3));

  render(&hittables, &textures, &world);
}

void checkered_spheres_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(5));
  TextureList textures = make_texture_list(&texture_arena);

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList world = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  TextureRef color_even = texture_list_add(&textures,
      make_solid_color_texture(Color(0.2f, 0.3f, 0.1f)));
  TextureRef color_odd = texture_list_add(&textures, 
      make_solid_color_texture(Color(0.9f, 0.9f, 0.9f)));
  TextureRef checker_texture = texture_list_add(&textures,
      make_checker_texture(0.32f, color_even, color_odd));

  Hittable sphere1 = make_hittable_sphere(Vec3(0.0f, -10.0f, 0.0f), 10.0f, make_lambertian(checker_texture));
  Hittable sphere2 = make_hittable_sphere(Vec3(0.0f, 10.0f, 0.0f), 10.0f, make_lambertian(checker_texture));

  hittable_ref_list_add(&world, hittable_list_add(&hittables, sphere1));
  hittable_ref_list_add(&world, hittable_list_add(&hittables, sphere2));

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
  settings.background_color = Vec3(0.7f, 0.8f, 1.0f);

  render_w_settings(&hittables, &textures, &world, &settings);
}

void earth_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(10));
  TextureList textures = make_texture_list(&texture_arena);

  TextureRef earth_texture = texture_list_add(&textures, make_image_texture("earthmap.jpg"));
  Material earth_surface = make_lambertian(earth_texture);

  Hittable globe = make_hittable_sphere(Vec3(0,0,0), 2.0f, earth_surface);

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList world = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  world_add_rotated(&hittables, &world, globe, 15);

  RenderSettings settings = g_default_render_settings;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.lookfrom = Vec3(0, 0, 12);
  settings.lookat = Vec3(0, 0, 0);
  settings.defocus_angle = 0;

  render_w_settings(&hittables, &textures, &world, &settings);
}

void perlin_spheres_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(5));
  TextureList textures = make_texture_list(&texture_arena);

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList world = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  PerlinNoise noise = make_perlin_noise();
  TextureRef noise_tex = texture_list_add(&textures, make_noise_texture(&noise, 4));
  Hittable ground = make_hittable_sphere(Vec3(0, -1000, 0), 1000, make_lambertian(noise_tex));
  Hittable sphere = make_hittable_sphere(Vec3(0, 2, 0), 2, make_lambertian(noise_tex));

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  world_add(&hittables, &world, ground);
  world_add(&hittables, &world, sphere);

  RenderSettings settings = g_default_render_settings;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.lookfrom = Vec3(13, 2, 3);
  settings.lookat = Vec3(0, 0, 0);
  settings.defocus_angle = 0;

  render_w_settings(&hittables, &textures, &world, &settings);
}

/*
void quads_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, sizeof(Texture) * 10);
  
  Hittable left_quad = make_hittable_quad(Vec3(-3,-2, 5), Vec3(0, 0,-4), Vec3(0, 4, 0), 
      make_lambertian_solid_color(&texture_arena, Vec3(1.0f, 0.2f, 0.2f)));
  Hittable back_quad = make_hittable_quad(Vec3(-2,-2, 0), Vec3(4, 0, 0), Vec3(0, 4, 0),
      make_lambertian_solid_color(&texture_arena, Vec3(0.2f, 1.0f, 0.2f)));
  Hittable right_quad = make_hittable_quad(Vec3(3,-2, 1), Vec3(0, 0, 4), Vec3(0, 4, 0),
      make_lambertian_solid_color(&texture_arena, Vec3(0.2f, 0.2f, 1.0f)));
  Hittable upper_quad = make_hittable_quad(Vec3(-2, 3, 1), Vec3(4, 0, 0), Vec3(0, 0, 4),
      make_lambertian_solid_color(&texture_arena, Vec3(1.0f, 0.5f, 0.0f)));
  Hittable lower_quad = make_hittable_quad(Vec3(-2,-3, 5), Vec3(4, 0, 0), Vec3(0, 0,-4),
      make_lambertian_solid_color(&texture_arena, Vec3(0.2f, 0.8f, 0.8f)));

  Hittable hittables[WORLD_SIZE]; // remember about the merge sort copy

  HittableList world = {0};
  world.objects = hittables;
  world.size = WORLD_SIZE;
  hittable_list_add(&world, left_quad);
  hittable_list_add(&world, back_quad);
  hittable_list_add(&world, right_quad);
  hittable_list_add(&world, upper_quad);
  hittable_list_add(&world, lower_quad);

  RenderSettings settings = {0};
  settings.img_ratio = 1.0f;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.max_bounces = 50;
  settings.vfov = 80;
  settings.lookfrom = Vec3(0, 0, 9);
  settings.lookat = Vec3(0, 0, 0);
  settings.view_up = Vec3(0, 1, 0);
  settings.defocus_angle = 0;
  settings.background_color = Vec3(0.7f, 0.8f, 1.0f);

  render_w_settings(&world, &settings);
}
*/

void simple_light_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(5));
  TextureList textures = make_texture_list(&texture_arena);

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList world = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  PerlinNoise noise = make_perlin_noise();
  TextureRef noise_tex = texture_list_add(&textures, make_noise_texture(&noise, 4));
  Hittable sphere1 = make_hittable_sphere(Vec3(0, -1000, 0), 1000, make_lambertian(noise_tex));
  Hittable sphere2 = make_hittable_sphere(Vec3(0, 2, 0), 2, make_lambertian(noise_tex));

  TextureRef sc = texture_list_add(&textures, make_solid_color_texture(Vec3(4, 4, 4)));
  Hittable light_quad =
    make_hittable_quad(Vec3(3, 1, -2), Vec3(2, 0, 0), Vec3(0, 2, 0), make_diffuse_light(sc));
  Hittable light_sphere = make_hittable_sphere(Vec3(0, 7, 0), 2, make_diffuse_light(sc));

  world_add(&hittables, &world, sphere1);
  world_add(&hittables, &world, sphere2);
  world_add(&hittables, &world, light_quad);
  world_add(&hittables, &world, light_sphere);

  RenderSettings settings = {0};
  settings.img_ratio = 16.0f / 9.0f;
  settings.img_width = 1080;
  settings.samples_per_pixel = 1000;
  settings.max_bounces = 50;
  settings.vfov = 20;
  settings.lookfrom = Vec3(26, 3, 6);
  settings.lookat = Vec3(0, 2, 0);
  settings.view_up = Vec3(0, 1, 0);

  render_w_settings(&hittables, &textures, &world, &settings);
}

void cornell_box_scene() {
  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(5));
  TextureList textures = make_texture_list(&texture_arena);

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList world = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  Material red = make_lambertian(texture_list_add(&textures,
        make_solid_color_texture(Vec3(0.65f, 0.05f, 0.05f))));
  Material white = make_lambertian(texture_list_add(&textures,
      make_solid_color_texture(Vec3(0.73f, 0.73f, 0.73f))));
  Material green = make_lambertian(texture_list_add(&textures,
      make_solid_color_texture(Vec3(0.12f, 0.45f, 0.15f))));
  TextureRef light_col = texture_list_add(&textures, make_solid_color_texture(Vec3(15, 15, 15)));
  Material light = make_diffuse_light(light_col);

  hittable_ref_list_add(&world,
        hittable_list_add(&hittables, 
          make_hittable_quad(Vec3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green)));
  hittable_ref_list_add(&world,
        hittable_list_add(&hittables, 
          make_hittable_quad(Vec3(0,0,0), Vec3(0,555,0), Vec3(0,0,555), red)));
  hittable_ref_list_add(&world,
        hittable_list_add(&hittables, 
          make_hittable_quad(Vec3(343, 554, 332), Vec3(-130,0,0), Vec3(0,0,-105), light)));
  hittable_ref_list_add(&world,
        hittable_list_add(&hittables, 
          make_hittable_quad(Vec3(0,0,0), Vec3(555,0,0), Vec3(0,0,555), white)));
  hittable_ref_list_add(&world,
        hittable_list_add(&hittables, 
          make_hittable_quad(Vec3(555,555,555), Vec3(-555,0,0), Vec3(0,0,-555), white)));
  hittable_ref_list_add(&world,
        hittable_list_add(&hittables, 
          make_hittable_quad(Vec3(0,0,555), Vec3(555,0,0), Vec3(0,555,0), white)));

#if 1
  Material metal = make_metal(Vec3(0.7f, 0.6f, 0.5f), 0.6f);
  world_add_rotated_translated(&hittables, 
      &world, 
      make_pyramid(&hittables, Vec3(0, 0, 0), Vec3(160, 0, 160), Vec3(100,400,100), red), 
      60,
      Vec3(200, 0, 150));

#else
  // BUG?: wall doesn't render without sorting when creating bvh
  world_add_translated(&hittables, 
      &world, 
      make_hittable_quad(Vec3(0, 0, 0), Vec3(100, 0, 0), Vec3(50, 50, 50), green), 
      Vec3(500, 200, 100));

  world_add_translated(&hittables, 
      &world, 
      make_hittable_quad(Vec3(0, 0, 0), Vec3(100, 0, 0), Vec3(50, 50, 50), green), 
      Vec3(100, 200, 100));

  world_add_translated(&hittables, 
      &world, 
      make_hittable_quad(Vec3(0, 0, 0), Vec3(100, 0, 0), Vec3(50, 50, 50), green), 
      Vec3(0, 200, 100));
#endif

#if 0
  HittableRef box1 = hittable_list_add(&hittables,
      make_box(&hittables, Vec3(130, 0, 65), Vec3(295, 165, 230), white));
  //box1 = hittable_list_add(&hittables, make_hittable_y_rotation(&hittables, box1, 15));
  hittable_ref_list_add(&world, box1);
#else
  HittableRef box1 = hittable_list_add(&hittables,
      make_box(&hittables, Vec3(0, 0, 0), Vec3(165, 330, 165), white));
  box1 = hittable_list_add(&hittables, make_hittable_y_rotation(&hittables, box1, 15));
  box1 = hittable_list_add(&hittables,
      make_hittable_translation(&hittables, box1, Vec3(265, 0, 295)));
  hittable_ref_list_add(&world, box1);
#endif

#if 0
  HittableRef box2 = hittable_list_add(&hittables,
      make_box(&hittables, Vec3(265, 0, 295), Vec3(430, 330, 460), white));
  //box2 = hittable_list_add(&hittables, make_hittable_y_rotation(&hittables, box2, -18));
  hittable_ref_list_add(&world, box2);
#else
  HittableRef box2 = hittable_list_add(&hittables,
      make_box(&hittables, Vec3(0, 0, 0), Vec3(165, 165, 165), white));
  box2 = hittable_list_add(&hittables, make_hittable_y_rotation(&hittables, box2, -18));
  box2 = hittable_list_add(&hittables,
      make_hittable_translation(&hittables, box2, Vec3(130, 0, -10)));
  hittable_ref_list_add(&world, box2);
#endif

  RenderSettings settings = {0};
  settings.img_ratio = 1.0f;
  settings.img_width = 600;
  settings.samples_per_pixel = 2000;
  settings.max_bounces = 50;
  settings.vfov = 40;
  settings.lookfrom = Vec3(278, 278, -800);
  settings.lookat = Vec3(278, 278, 0);
  settings.view_up = Vec3(0, 1, 0);
  settings.defocus_angle = 0;

  render_w_settings(&hittables, &textures, &world, &settings);
}

#if 0
void final_scene(World *world) {
  Material ground_mat = add_mat_lamb_sc(world, Vec3(0.48f, 0.83f, 0.53f));

  i32 boxes_per_side = 20;
  for (i32 i = 0; i < boxes_per_side; ++i) {
    for (i32 j = 0; j < boxes_per_side; ++j) {
      f32 w = 100;
      f32 x0 = -1000.0f + i * w;
      f32 z0 = -1000.0f + j * w;
      f32 y0 = 0;
      f32 x1 = x0 + w;
      f32 y1 = random_f32_bound(1, 101);
      f32 z1 = z0 + w;

      add_box(world, Vec3(x0, y0, z0), Vec3(x1, y1, z1), ground_mat);
    }
  }

  Material light = add_mat_light_sc(world, Vec3(7, 7, 7));
  add_quad(world, Vec3(123, 554, 147), Vec3(300, 0, 0), Vec3(0, 0, 265), light);

  vec3 center1 = { 400, 400, 200 };
  vec3 center2 = vec3_add(center1, Vec3(30, 0, 0));
  Material sphere_mat = add_mat_lamb_sc(world, Vec3(0.7f, 0.3f, 0.1f));
  add_mov_sphere(world, center1, center2, 50, sphere_mat);

  add_sphere(world, Vec3(260, 140, 45), 40, add_mat_dielectric(world, 1.5f));
  add_sphere(world, Vec3(0, 150, 145), 50, add_mat_metal_sc(world, Vec3(0.8f, 0.8f, 0.9f), 1.0f));

  Material earth_mat = add_mat_img("earthmap.jpg");
  add_sphere(world, Vec3(400, 200, 400), 100, earth_mat);
  Material perlin_noise = add_mat_lamb_noise(world, 0.2f);

  Material white = add_mat_lamb_sc(world, Vec3(0.73f, 0.73f, 0.73f));
  i32 ns = 1000;
  for (i32 i = 0; i < ns; ++i) {
    add_sphere(vec3_random_bound(0, 165), 10, white);
  }

  RenderSettings settings = {0};
  settings.img_ratio = 1.0f;
  settings.img_width = 400;
  settings.samples_per_pixel = 100;
  settings.max_bounces = 50;
  settings.vfov = 40.0f;
  settings.lookfrom = Vec3(478, 278, -600);
  settings.lookat = Vec3(278, 278, 0);
  settings.view_up = Vec3(0.0f, 0.1f, 0.0f);
  settings.defocus_angle = 0.0f;
  
  render(world, settings);
}
#endif

i32 main() {
  // vec3_to_unit_vec_test();

  //bouncing_spheres_scene();
  //checkered_spheres_scene();
  //earth_scene();
  //perlin_spheres_scene();
  // quads_scene();
  // simple_light_scene();
  cornell_box_scene();
}
