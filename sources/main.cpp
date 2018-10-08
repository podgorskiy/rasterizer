#include <stdio.h>
#include <stdint.h>

#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Context.h"

namespace py = pybind11;

template<typename T>
uint8_t* TOUINT8(T* data, int size)
{
	uint8_t* out = new uint8_t[size];
	for (int i = 0; i < size; ++i)
	{
		T x = data[i];
		if (x < 0.0f) x = 0.0f;
		if (x > 1.0f) x = 1.0f;
		out[i] = (uint8_t)(x * 255.0f);
	}
	return out;
}

/*
) / sizeof(data_x2[0]);

int size_x = 224;
int size_y = 224;
int size_z = 2;
int max_vector = 4096;

Context ctx = InitContext(max_vector, size_x, size_y, size_z);

memcpy(ctx.vector_host + max_vector * 0, data_x, poin_count * sizeof(float));
memcpy(ctx.vector_host + max_vector * 1, data_y, poin_count * sizeof(float));
memcpy(ctx.vector_host + max_vector * 2, data_x2, poin_count2 * sizeof(float));
memcpy(ctx.vector_host + max_vector * 3, data_y2, poin_count2 * sizeof(float));

Render(&ctx, 2);

uint8_t* data = TOUINT8(ctx.raster_host, size_z * size_x * size_y);

stbi_write_png("sample0.png", size_x, size_y, 1, data, size_x);
stbi_write_png("sample1.png", size_x, size_y, 1, data + size_x * size_y, size_x);
*/

class Rasterizer
{
public:
	Rasterizer& operator=(const Context&) = delete;
	Rasterizer(const Context&) = delete;

	Rasterizer(int size_x, int size_y, int size_z)
	{
		int max_vector = 4096;
		ctx = InitContext(max_vector, size_x, size_y, size_z);
	}

	~Rasterizer()
	{
		FreeContext(&ctx);
	}

	py::array_t<float, py::array::c_style> Render(std::vector<std::vector<py::array_t<float, py::array::c_style> > > x)
	{
		for (int i = 0, l = x.size(); i < l; ++i)
		{
			float* dst_x = ctx.vector_host + ctx.stride * 2 * i;
			float* dst_y = ctx.vector_host + ctx.stride * (2 * i + 1);

			for (int j = 0, m = x[i].size(); j < m; ++j)
			{
				py::array_t<float, py::array::c_style>& vec = x[i][j];

				auto p = vec.unchecked<2>();
				int w = (int)p.shape(1);
				int h = (int)p.shape(0);

				const float* __restrict data_x = p.data(0, 0);
				const float* __restrict data_y = p.data(1, 0);

				/*
				for (int k = 0; k < w; ++k)
				{
					printf("%f ", data_x[k]);
				}
				printf("\n");
				for (int k = 0; k < w; ++k)
				{
					printf("%f ", data_y[k]);
				}
				printf("\n");
				*/

				memcpy(dst_x, data_x, w * sizeof(float));
				memcpy(dst_y, data_y, w * sizeof(float));
				
				if (j != m-1)
				{
					dst_x[w] = dst_x[w - 1];
					dst_y[w] = dst_y[w - 1];

					++dst_x;
					++dst_y;
				}

				dst_x += w;
				dst_y += w;
			}
			*dst_x = -1;
			*dst_y = -1;
		}

		::Render(&ctx, x.size());

		return py::array_t<float>(
			std::vector<uint64_t>{ (uint64_t)x.size(), (uint64_t)ctx.size_y, (uint64_t)ctx.size_x },
			ctx.raster_host);
	}


private:
	Context ctx;
};


PYBIND11_MODULE(rasterizer, m) {
	m.doc() = "";

	py::class_<Rasterizer>(m, "Rasterizer")
		.def(py::init<int, int, int>())
		.def("Render", &Rasterizer::Render);
}
