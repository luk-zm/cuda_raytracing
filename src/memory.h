#ifndef MEMORY_H
#define MEMORY_H

#define PTR_TO_INT(p) (u64)((u8*)p - (u8*)0)
#define INT_TO_PTR(n) (u8*)((u8*)0 + (n))

typedef struct {
  u8 *data;
  u64 current_pos;
  u64 size;
} MemArena;

//NOTE: alignment must be a power of 2
__device__ __host__
inline u64 align_up(u64 p, u64 alignment) {
  return (p + alignment - 1) & ~(alignment - 1);
}

__device__ __host__
inline u8 *align_ptr_up(u8 *p, u64 alignment) {
  return INT_TO_PTR(align_up(PTR_TO_INT(p), alignment));
}

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


#endif
