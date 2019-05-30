#pragma once

#include <vector>

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
	float * voxelGridWeight	/*output buffer*/);

void SaveVoxelGrid2SurfacePointCloud(
	const std::string &file_name,
	int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
	float voxel_size,
	float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
	float * voxel_grid_TSDF,
	float * voxel_grid_weight,
	float tsdf_thresh,
	float weight_thresh);

std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N);

bool invert_matrix(const float m[16], float invOut[16]);

// read in raw depth image in row-major
void ReadRawDepth(std::string filename, int H, int W, float * depth);

void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]);