#ifndef AABB_H
#define AABB_H
#include "utility.h"
#include "vec3.h"

typedef struct {
  f32 x0, x1;
  f32 y0, y1;
  f32 z0, z1;
} Aabb;

b32 aabb_hit(Aabb bbox, vec3 ray_origin, vec3 ray_direction, f32 ray_t0, f32 ray_t1);

#endif
