#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <curand_kernel.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "utility.h"
#include "vec3.h"
#include "perlin_noise.h"
#include "memory.h"
#include "platform.h"

// unity build
#include "perlin_noise.cpp"
#include "platform.cpp"
#include "raytracing.cu"


#define WORLD_SIZE 8000

#define SCENE(x) x(&world, &settings)
#define DEFSCENE(name) void name(World *world, RenderSettings *settings)

i32 main(int argc, char *argv[]) {
  i32 scene_select = 0;
  if (argc > 1) {
    scene_select = atoi(argv[1]);
  }

  MemArena texture_arena = {0};
  mem_arena_alloc(&texture_arena, MB(10));
  TextureList textures = make_texture_list(&texture_arena);

  MemArena hittable_ref_arena = {0};
  mem_arena_alloc(&hittable_ref_arena, sizeof(HittableRef) * WORLD_SIZE);
  HittableRefList bvh_include = make_hittable_ref_list(&hittable_ref_arena, WORLD_SIZE);

  Hittable hittables_data[WORLD_SIZE]; // remember about the merge sort copy

  HittableList hittables = {0};
  hittables.objects = hittables_data;
  hittables.size = WORLD_SIZE;

  World world = { hittables, textures, bvh_include };
  RenderSettings settings = g_default_render_settings;

  switch (scene_select) {
    case 0:
    SCENE(fv1);
    break;
    case 1:
    SCENE(cornell_box_scene);
    break;
    case 2:
    SCENE(bouncing_spheres_scene);
    break;
    case 3:
    SCENE(final_scene);
    break;
    case 4:
    SCENE(fv2);
    break;
    case 5:
    SCENE(fv3);
    break;
    case 6:
    SCENE(fv4);
    break;
    case 7:
    SCENE(fv5);
    break;
    case 8:
    SCENE(refract1);
    break;
    case 9:
    SCENE(refract2);
    break;
    case 10:
    SCENE(refract3);
    break;
    case 11:
    SCENE(refract4);
    break;
    case 12:
    SCENE(refract5);
    break;
    case 13:
    SCENE(motion_blur_scene);
    break;
    case 14:
    SCENE(light_test_scene);
    break;
    case 15:
    SCENE(checkered_spheres_scene);
    break;
    case 16:
    SCENE(earth_scene);
    break;
    case 17:
    SCENE(perlin_spheres_scene);
    break;
    case 18:
    SCENE(quads_scene);
    break;
    case 19:
    SCENE(simple_light_scene);
    break;
    default:
    printf("Unknown scene\n");
    return 1;
    break;
  }
  if (argc > 2) {
    if (argv[2][0] == 'b')
      settings.max_bounces = atoi(argv[2] + 1);
    else if (argv[2][0] == 's')
      settings.samples_per_pixel = atoi(argv[2] + 1);
  }
  if (argc > 3) {
    if (argv[3][0] == 'b')
      settings.max_bounces = atoi(argv[3] + 1);
    else if (argv[3][0] == 's')
      settings.samples_per_pixel = atoi(argv[3] + 1);
  }
  render(&world, &settings);
}
