#include "kernels.h"

#include <cmath>
#include <string>
#include <fstream>
#include <sstream>


void Integrate(
	float * camIntrin,	/*camera intrinsics (3x3 float matrix)*/
	float * cam2Base,	/*current camera pose w.r.t to base coordinate (4x4 float matrix)*/
	float * depthImage,	/*current camera depth image*/
	int width, int height,	/*depth image width and height (in pixels)*/
	int voxelGridDimX,
	int voxelGridDimY,
	int voxelGridDimZ,	/*voxel grid dimensions*/
	float voxelGridOriginX,
	float voxelGridOriginY,
	float voxelGridOriginZ, /*voxel grid origin in base coordinate*/
	float voxelSize, /*size of each cubic voxel*/
	float truncMargin,	/*Truncation margin for SDF*/
	float * voxelGridTSDF, /*output buffer*/
	float * voxelGridWeight	/*output buffer*/) 
{
	for (int pt_grid_z = 0; pt_grid_z < voxelGridDimZ; pt_grid_z++)
	{
		for (int pt_grid_y = 0; pt_grid_y < voxelGridDimY; pt_grid_y++)
		{
#pragma omp parallel for
			for (int pt_grid_x = 0; pt_grid_x < voxelGridDimX; pt_grid_x++) 
			{
				// Convert voxel center from grid coordinates (int) to base frame camera coordinates (float)
				float pt_base_x = voxelGridOriginX + pt_grid_x * voxelSize;
				float pt_base_y = voxelGridOriginY + pt_grid_y * voxelSize;
				float pt_base_z = voxelGridOriginZ + pt_grid_z * voxelSize;

				// Convert from base coordinates to current frame camera coordinates
				float tmp_pt[3];
				tmp_pt[0] = pt_base_x - cam2Base[0 * 4 + 3];
				tmp_pt[1] = pt_base_y - cam2Base[1 * 4 + 3];
				tmp_pt[2] = pt_base_z - cam2Base[2 * 4 + 3];
				float pt_cam_x = cam2Base[0 * 4 + 0] * tmp_pt[0] + cam2Base[1 * 4 + 0] * tmp_pt[1] + cam2Base[2 * 4 + 0] * tmp_pt[2];
				float pt_cam_y = cam2Base[0 * 4 + 1] * tmp_pt[0] + cam2Base[1 * 4 + 1] * tmp_pt[1] + cam2Base[2 * 4 + 1] * tmp_pt[2];
				float pt_cam_z = cam2Base[0 * 4 + 2] * tmp_pt[0] + cam2Base[1 * 4 + 2] * tmp_pt[1] + cam2Base[2 * 4 + 2] * tmp_pt[2];

				if (pt_cam_z <= 0)
				{
					// the camera must be always outside the voxel grid !!!
					continue;
				}

				// project voxel grid 3D position onto depth image pixel 
				int pt_pix_x = std::roundf(camIntrin[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + camIntrin[0 * 3 + 2]);
				int pt_pix_y = std::roundf(camIntrin[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + camIntrin[1 * 3 + 2]);
				
				if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
				{
					// skip voxels that cannot be seen from current camera
					continue;
				}

				float depth_val = depthImage[pt_pix_y * width + pt_pix_x];

				// we can filter out some illegal depth values
				// this is useful for pre-processing depth images
				//if (depth_val <= 0 || depth_val > 6)
				if(depth_val <= 0)
				{
					continue;
				}

				// compute the SDF value
				float diff = depth_val - pt_cam_z;
				// truncate the SDF value
				if (diff <= -truncMargin)
				{
					continue;
				}
				float dist = std::fmin(1.0f, diff / truncMargin);
				// integrate with previous values
				int volume_idx = pt_grid_z * voxelGridDimY * voxelGridDimX + pt_grid_y * voxelGridDimX + pt_grid_x;
				float weight_old = voxelGridWeight[volume_idx];
				float weight_new = weight_old + 1.0f;
				voxelGridWeight[volume_idx] = weight_new;
				voxelGridTSDF[volume_idx] = (voxelGridTSDF[volume_idx] * weight_old + dist) / weight_new;
			}
		}
	}
	// over
	return;
}

std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N)
{
	std::vector<float> matrix;
	FILE *fp = fopen(filename.c_str(), "r");
	for (int i = 0; i < M * N; i++) 
	{
		float tmp;
		int iret = fscanf(fp, "%f", &tmp);
		matrix.push_back(tmp);
	}
	fclose(fp);
	return matrix;
}

bool invert_matrix(const float m[16], float invOut[16]) 
{
	float inv[16], det;
	int i;
	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

void ReadRawDepth(std::string filename, int H, int W, float * depth)
{
	std::ifstream ins;
	ins.open(filename, std::ios::binary);
	ins.read((char*)depth, sizeof(float)*H*W);
	ins.close();
	return;
}

void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]) 
{
	mOut[0] = m1[0] * m2[0] + m1[1] * m2[4] + m1[2] * m2[8] + m1[3] * m2[12];
	mOut[1] = m1[0] * m2[1] + m1[1] * m2[5] + m1[2] * m2[9] + m1[3] * m2[13];
	mOut[2] = m1[0] * m2[2] + m1[1] * m2[6] + m1[2] * m2[10] + m1[3] * m2[14];
	mOut[3] = m1[0] * m2[3] + m1[1] * m2[7] + m1[2] * m2[11] + m1[3] * m2[15];

	mOut[4] = m1[4] * m2[0] + m1[5] * m2[4] + m1[6] * m2[8] + m1[7] * m2[12];
	mOut[5] = m1[4] * m2[1] + m1[5] * m2[5] + m1[6] * m2[9] + m1[7] * m2[13];
	mOut[6] = m1[4] * m2[2] + m1[5] * m2[6] + m1[6] * m2[10] + m1[7] * m2[14];
	mOut[7] = m1[4] * m2[3] + m1[5] * m2[7] + m1[6] * m2[11] + m1[7] * m2[15];

	mOut[8] = m1[8] * m2[0] + m1[9] * m2[4] + m1[10] * m2[8] + m1[11] * m2[12];
	mOut[9] = m1[8] * m2[1] + m1[9] * m2[5] + m1[10] * m2[9] + m1[11] * m2[13];
	mOut[10] = m1[8] * m2[2] + m1[9] * m2[6] + m1[10] * m2[10] + m1[11] * m2[14];
	mOut[11] = m1[8] * m2[3] + m1[9] * m2[7] + m1[10] * m2[11] + m1[11] * m2[15];

	mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8] + m1[15] * m2[12];
	mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9] + m1[15] * m2[13];
	mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
	mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

void SaveVoxelGrid2SurfacePointCloud(
	const std::string &file_name, 
	int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
	float voxel_size, 
	float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
	float * voxel_grid_TSDF, 
	float * voxel_grid_weight,
	float tsdf_thresh, 
	float weight_thresh)
{
	std::vector<float> output_buffer;
	// Create point cloud content for ply file
	for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++) {

		// If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
		if (std::abs(voxel_grid_TSDF[i]) < tsdf_thresh && voxel_grid_weight[i] > weight_thresh) {

			// Compute voxel indices in int for higher positive number range
			int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
			int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
			int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

			// Convert voxel indices to float, and save coordinates to ply file
			float pt_base_x = voxel_grid_origin_x + (float)x * voxel_size;
			float pt_base_y = voxel_grid_origin_y + (float)y * voxel_size;
			float pt_base_z = voxel_grid_origin_z + (float)z * voxel_size;
			// store in buffer
			output_buffer.push_back(pt_base_x);
			output_buffer.push_back(pt_base_y);
			output_buffer.push_back(pt_base_z);
		}
	}
	
	// write to the .ply file
	std::ostringstream header_writer;
	header_writer
		<< "ply" << std::endl
		<< "format binary_little_endian 1.0" << std::endl
		<< "element vertex " << output_buffer.size() / 3 << std::endl
		<< "property float x" << std::endl
		<< "property float y" << std::endl
		<< "property float z" << std::endl
		<< "end_header" << std::endl;
	std::string header = header_writer.str();

	std::ofstream outs;
	outs.open(file_name, std::ios::binary);
	outs.write(header.data(), header.size());
	outs.write((char*)output_buffer.data(), output_buffer.size() * sizeof(float));
	outs.close();

	// over
	return;
}
