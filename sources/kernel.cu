#include <cuda.h>
#include "helper_math.h"
#include "Context.h"
#include <stdio.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ float2 LineClosestPoint(float2 a, float2 b, float2 p)
{
	float2 ap = p - a;
	float2 ab_dir = b - a;
	float dot = ap.x * ab_dir.x + ap.y * ab_dir.y;
	if (dot < 0.0f)
		return a;
	float ab_len_sqr = ab_dir.x * ab_dir.x + ab_dir.y * ab_dir.y;
	if (dot > ab_len_sqr)
		return b;
	return a + ab_dir * dot / ab_len_sqr;
}

__global__ void kernel(float* A, float* v, int stride) 
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	float2 p;
	p.x = float(j) * 256.0f / 224;// / gridDim.x / blockDim.x;
	p.y = float(i) * 256.0f / 224;// / gridDim.y / blockDim.y;
		
	int it = 0;
	int offsetx = stride * 2 * k;
	int offsety = stride * (2 * k + 1);

	float min_dist = 1e6f;
	for (;it < stride; ++it)
	{
		float2 p0;
		p0.x = v[offsetx + it];
		p0.y = v[offsety + it];
		float2 p1;
		p1.x = v[offsetx + it + 1];
		p1.y = v[offsety + it + 1];

		if (p1.x == p0.x && p0.y == p1.y)
		{
			it += 1;
			continue;
		}
		if (p1.x <0 && p1.y <0)
		{
			break;
		}

		float2 x = LineClosestPoint(p0, p1, p);
		float2 d = x - p;
		float distance = hypotf(d.x, d.y);
		if (min_dist > distance)
		{
			min_dist = distance;
		}
	}

	A[j + i * gridDim.x * blockDim.x + k * gridDim.x * blockDim.x * gridDim.y * blockDim.y] = expf(-min_dist * min_dist / 2.0f);
}


DeviceGuard::DeviceGuard(int device)
{
	gpuErrchk(cudaGetDevice(&backup_device));
	gpuErrchk(cudaSetDevice(device));
}

DeviceGuard::~DeviceGuard()
{
	gpuErrchk(cudaSetDevice(backup_device));
}

Context InitContext(int stride, int size_x, int size_y, int size_z, int device)
{
	Context ctx;
	ctx.device = device;
	DeviceGuard cuda(ctx.device);

	gpuErrchk(cudaMalloc(&ctx.vector_array, 2 * stride * size_z * sizeof(float)));
	ctx.size_x = size_x;
	ctx.size_y = size_y;
	ctx.size_z = size_z;
	ctx.stride = stride;

	gpuErrchk(cudaMalloc(&ctx.raster_array, size_z * size_x * size_y * sizeof(float)));
	ctx.raster_host = new float[size_z * size_x * size_y];
	ctx.vector_host = new float[2 * stride * size_z];

	return ctx;
}

void FreeContext(Context* ctx)
{
	DeviceGuard cuda(ctx->device);

	gpuErrchk(cudaFree(ctx->vector_array));
	gpuErrchk(cudaFree(ctx->raster_array));
	delete[] ctx->raster_host;
	delete[] ctx->vector_host;
}

void Render(Context* ctx, int count)
{
	DeviceGuard cuda(ctx->device);

	int threads_in_block_x = 8;
	int threads_in_block_y = 8;

	int blocks_x = ctx->size_x / threads_in_block_x;
	int blocks_y = ctx->size_y / threads_in_block_y;

	gpuErrchk(cudaMemcpy(ctx->vector_array, ctx->vector_host, 2 * ctx->stride * count * sizeof(float), cudaMemcpyHostToDevice));

	kernel<<<dim3(blocks_x, blocks_y, count), dim3(threads_in_block_x, threads_in_block_y, 1)>>>(ctx->raster_array, ctx->vector_array, ctx->stride);

	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(ctx->raster_host, ctx->raster_array, count * ctx->size_x * ctx->size_y * sizeof(float), cudaMemcpyDeviceToHost));
}
