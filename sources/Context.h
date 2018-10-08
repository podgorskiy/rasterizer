#pragma once

struct Context
{
	float* raster_array = nullptr;
	float* vector_array = nullptr;
	float* raster_host = nullptr;
	float* vector_host = nullptr;
	int size_x;
	int size_y;
	int size_z;
	int stride;
};

Context InitContext(int stride, int size_x, int size_y, int size_z);

void FreeContext(Context* ctx);

void Render(Context* ctx, int count);
