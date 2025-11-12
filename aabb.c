#include "aabb.h"

b32 aabb_hit(Aabb bbox, vec3 ray_origin, vec3 ray_direction, f32 ray_t0, f32 ray_t1) {
  f32 t0_x = (bbox.x0 - ray_origin.x) / ray_direction.x;
  f32 t1_x = (bbox.x1 - ray_origin.x) / ray_direction.x;

  interval_sort(&t0_x, &t1_x);
  if (!interval_overlap(t0_x, t1_x, ray_t0, ray_t1))
    return 0;

  f32 t0_y = (bbox.y0 - ray_origin.y) / ray_direction.y;
  f32 t1_y = (bbox.y1 - ray_origin.y) / ray_direction.y;

  interval_sort(&t0_y, &t1_y);
  if (!interval_overlap(t0_x, t1_x, t0_y, t1_y))
    return 0;

  f32 t0_z = (bbox.z0 - ray_origin.z) / ray_direction.z;
  f32 t1_z = (bbox.z1 - ray_origin.z) / ray_direction.z;

  interval_sort(&t0_z, &t1_z);
  return interval_overlap(t0_y, t1_y, t0_z, t1_z);
}

