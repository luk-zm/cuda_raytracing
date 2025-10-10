#include <math.h>

typedef struct {
	float x;
	float y;
	float z;
} vec3;

float vec_length(vec3 v) {
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

vec3 unit_vec(vec3 v) {
	float v_len = vec_length(v);
	vec3 result = {v.x / v_len, v.y / v_len, v.z / v_len};
	return result;
}

int main() {
	vec3 camera_pos = { 0.0f, 0.0f, 0.0f };
	int img_height = 400;
	double img_ratio = 16.0/9;
	int img_width = 400 * img_ratio;
	
	float viewport_height = 2.0f;
	double viewport_ratio = (double)img_height /  img_width;
	float viewport_width = viewport_height * viewport_ratio;
	
	float camera_viewport_dist = 1.0f;
	
	vec3 viewport_q = camera_pos;
	viewport_q.z += camera_viewport_dist;
	viewport_q.x -= viewport_width / 2;
	viewport_q.y += viewport_height / 2;
	
	vec3 p00 = viewport_q;
	float pixel_dist = viewport_height / img_height;
	
	p00.x += pixel_dist;
	p00.y -= pixel_dist;
	
	vec3 rays_directions[img_height][img_width];
	rays_directions[0][0] = p00;
	
	for (int i = 0; i < img_height; ++i) {
		for (int j = 1; j < img_width; ++j) {
			p00.x += pixel_dist;
			rays_directions[i][j] = p00;
			rays_directions[i][j].x -= camera_pos.x;
			rays_directions[i][j].y -= camera_pos.y;
			rays_directions[i][j].z -= camera_pos.z;
		}
        p00.y -= pixel_dist;
	}
	
	vec3 rays_positions[img_height][img_width];
	
	// tracing
	for (int i = 0; i < img_height; ++i) {
		for (int j = 0; j < img_width; ++j) {
			for (float t = 0.0f; t < 10.0f; t += 0.01f) {
				if (!is_crossing(rays_positions[i][j]))
					rays_positions[i][j] = camera_pos + t * rays_directions[i][j]; // vec3 math not implemented
			}
		}
	}
	
	// check colors at intersections
}
	
