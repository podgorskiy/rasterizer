#pragma once

class DeviceGuard
{
public:
	DeviceGuard& operator=(const DeviceGuard&) = delete;
	DeviceGuard(const DeviceGuard&) = delete;

	DeviceGuard(int device);
	~DeviceGuard();
private:
	int backup_device;
};

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
	int device;
};

Context InitContext(int stride, int size_x, int size_y, int size_z, int device);

void FreeContext(Context* ctx);

void Render(Context* ctx, int count);
