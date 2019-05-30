#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "kernels.h"



// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char * argv[]) 
{
	// Location of camera intrinsic file
	std::string camIntrinFile = "data\\camera-intrinsics.txt";

	// Location of folder containing depth frames and camera pose files
	std::string data_path = "data\\rgb-frames";
	int base_frame_idx = 150;
	int first_frame_idx = 150;
	float num_frames = 50;

	const int width = 640;
	const int height = 480;
	float *depth_im = new float[width * height];

	float camIntrin[3 * 3];
	float base2World[4 * 4];
	float cam2Base[4 * 4];
	float cam2World[4 * 4];

	// Voxel grid parameters (change these to change voxel grid resolution, etc.)
	float voxelGridOriginX = -1.5f; // Location of voxel grid origin in base frame camera coordinates
	float voxelGridOriginY = -1.5f;
	float voxelGridOriginZ = 0.5f;
	float voxelSize = 0.006f;
	float truncMargin = voxelSize * 5;
	int voxelGridDimX = 500;
	int voxelGridDimY = 500;
	int voxelGridDimZ = 500;

	// Manual parameters
	if (argc > 1) 
	{
		camIntrinFile = argv[1];
		data_path = argv[2];
		base_frame_idx = atoi(argv[3]);
		first_frame_idx = atoi(argv[4]);
		num_frames = atof(argv[5]);
		voxelGridOriginX = atof(argv[6]);
		voxelGridOriginY = atof(argv[7]);
		voxelGridOriginZ = atof(argv[8]);
		voxelSize = atof(argv[9]);
		truncMargin = atof(argv[10]);
	}

	// Read camera intrinsics
	std::vector<float> camIntrinVec = LoadMatrixFromFile(camIntrinFile, 3, 3);
	std::copy(camIntrinVec.begin(), camIntrinVec.end(), camIntrin);

	// Read base frame camera pose
	std::ostringstream base_frame_prefix;
	base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
	std::string base2world_file = data_path + "\\frame-" + base_frame_prefix.str() + ".pose.txt";
	std::vector<float> base2WorldVec = LoadMatrixFromFile(base2world_file, 4, 4);
	std::copy(base2WorldVec.begin(), base2WorldVec.end(), base2World);

	// Invert base frame camera pose to get world-to-base frame transform 
	float base2world_inv[16];
	invert_matrix(base2World, base2world_inv);

	// Initialize voxel grid
	float * voxel_grid_TSDF = new float[voxelGridDimX * voxelGridDimY * voxelGridDimZ];
	float * voxel_grid_weight = new float[voxelGridDimX * voxelGridDimY * voxelGridDimZ];
	for (int i = 0; i < voxelGridDimX * voxelGridDimY * voxelGridDimZ; i++)
	{
		// initialize voxel grid to a value lgt threshold !!!
		voxel_grid_TSDF[i] = 1.0f;
	}
	memset(voxel_grid_weight, 0, sizeof(float) * voxelGridDimX * voxelGridDimY * voxelGridDimZ);

	// Loop through each depth frame and integrate TSDF voxel grid
	for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx) 
	{
		std::ostringstream curr_frame_prefix;
		curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

		// Read current frame depth
		std::string depth_im_file = data_path + "\\frame-" + curr_frame_prefix.str() + ".depth";
		ReadRawDepth(depth_im_file, height, width, depth_im);

		// Read base frame camera pose
		std::string cam2world_file = data_path + "\\frame-" + curr_frame_prefix.str() + ".pose.txt";
		std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
		std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2World);

		// Compute relative camera pose (camera-to-base frame)
		multiply_matrix(base2world_inv, cam2World, cam2Base);

		std::cout << "Fusing: " << depth_im_file << std::endl;

		Integrate(camIntrin, cam2Base, depth_im,
			width, height, 
			voxelGridDimX, voxelGridDimY, voxelGridDimZ,
			voxelGridOriginX, voxelGridOriginY, voxelGridOriginZ, 
			voxelSize, 
			truncMargin,
			voxel_grid_TSDF, voxel_grid_weight);
	}

	// Compute surface points from TSDF voxel grid and save to point cloud .ply file
	std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
	SaveVoxelGrid2SurfacePointCloud("tsdf.ply", 
		voxelGridDimX, voxelGridDimY, voxelGridDimZ,
		voxelSize, 
		voxelGridOriginX, voxelGridOriginY, voxelGridOriginZ,
		voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

	// Save TSDF voxel grid and its parameters to disk as binary file (float array)
	/*std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
	std::string voxel_grid_saveto_path = "tsdf.bin";
	std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
	float voxel_grid_dim_xf = (float)voxelGridDimX;
	float voxel_grid_dim_yf = (float)voxelGridDimY;
	float voxel_grid_dim_zf = (float)voxelGridDimZ;
	outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
	outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
	outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
	outFile.write((char*)&voxelGridOriginX, sizeof(float));
	outFile.write((char*)&voxelGridOriginY, sizeof(float));
	outFile.write((char*)&voxelGridOriginZ, sizeof(float));
	outFile.write((char*)&voxelSize, sizeof(float));
	outFile.write((char*)&truncMargin, sizeof(float));
	for (int i = 0; i < voxelGridDimX * voxelGridDimY * voxelGridDimZ; i++)
	{
		outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
	}
	outFile.close();*/

	// release memory
	delete[] depth_im;
	delete[] voxel_grid_TSDF;
	delete[] voxel_grid_weight;

	return 0;
}