#include <iostream>
#include <random>
#include "laplace.h"
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/writeOBJ.h>
#include <igl/file_exists.h>
#include <imgui/imgui.h>
#include <Eigen/SVD>


class windowStuff{
public:

	std::string pathMesh1 = "../data/curvatures/lilium_s.obj";

	windowStuff() :Noise(0.0), k(1), task(0){
		//initial vertices and faces
		if (!igl::file_exists(pathMesh1)){
			std::cout << "Initial mesh has incorrect Path! Exiting...";
			exit(1);
		}
	}
	~windowStuff() {}

	Eigen::MatrixXd V1;
	Eigen::MatrixXi F1;
	Eigen::VectorXd meanCurv;
	Eigen::VectorXd gausCurv;

    int mesh = 0;
    int task;
    int k;
	int numIters = 100;
	float Noise;
	float lambda = 1e-7;
    

	void reset(igl::opengl::glfw::Viewer& viewer)
	{
		static std::default_random_engine generator;
		viewer.data().clear();
        viewer.data().show_lines = 0;
        viewer.data().show_overlay_depth = 1;

		if (task == 1){
			std::cout << "Task 1.1: Uniform Laplace (Mean)" << std::endl;
			meanCurv = getH(V1,F1,false);
			meanCurv = 100 * meanCurv.array() / (meanCurv.maxCoeff() - meanCurv.minCoeff());
			viewer.data().set_mesh(V1, F1);
			viewer.data().set_colors(meanCurv);
			viewer.core().align_camera_center(V1, F1);
		}
		else if(task == 2){
			std::cout << "Task 1.2: Gaussian Curvature" << std::endl;
			gausCurv = estimateGaussCurv(V1, F1);
			gausCurv = 100 * gausCurv.array() / (gausCurv.maxCoeff() - gausCurv.minCoeff());
            viewer.data().set_mesh(V1, F1);
            viewer.data().set_colors(gausCurv);
            viewer.core().align_camera_center(V1, F1);
        }
		else if(task == 3){
			std::cout << "Task 3: Non-Uniform Laplace (Cotangent Mean)" << std::endl;
			meanCurv = getH(V1,F1,true);
			meanCurv = 100 * meanCurv.array() / (meanCurv.maxCoeff() - meanCurv.minCoeff());
			viewer.data().set_mesh(V1, F1);
			viewer.data().set_colors(meanCurv);
			viewer.core().align_camera_center(V1, F1);
		}
        else if(task == 4){
			std::cout << "Task 4: Mesh Reconstruction" << std::endl;
			Eigen::MatrixXd newMesh;
			newMesh = meshReconstruction(V1,F1,k);
			viewer.data().set_mesh(newMesh, F1);
			viewer.core().align_camera_center(newMesh, F1);
        }
		else if(task == 5){
			std::cout << "Task 5: Explicit Laplace Smoothing" << std::endl;
			V1 = explicitSmoothing(V1,F1,lambda,numIters);
			viewer.data().set_mesh(V1, F1);
			viewer.core().align_camera_center(V1, F1);
		}
		else if(task == 6){
			std::cout << "Task 6: Implicit Laplace Smoothing" << std::endl;
			V1 = implicitSmoothing(V1,F1,lambda,numIters);
			viewer.data().set_mesh(V1, F1);
			viewer.core().align_camera_center(V1, F1);
		}
		else if(task == 7){
			std::cout << "Task 7: Add Noise" << std::endl;
			V1 = addNoise(V1,Noise);
			viewer.data().set_mesh(V1, F1);
			viewer.core().align_camera_center(V1, F1);
		}
		else{
			std::cout << "Reset" << std::endl;
            switch(mesh){
                default:
                    pathMesh1 = "../data/curvatures/lilium_s.obj";
                    break;
                case 1:
                    pathMesh1 = "../data/curvatures/plane.obj";
                    break;
                case 2:
                    pathMesh1 = "../data/decompose/armadillo.obj";
                    break;
                case 3:
                    pathMesh1 = "../data/smoothing/fandisk_ns.obj";
                    break;
                case 4:
                    pathMesh1 = "../data/smoothing/plane_ns.obj";
                    break;
            }

			igl::readOBJ(pathMesh1, V1, F1);
            viewer.data().set_mesh(V1, F1);
			viewer.core().align_camera_center(V1, F1);
			viewer.data().show_overlay_depth = 1;
			viewer.data().show_overlay = 1;
		}
	}
private:
};


windowStuff viewerData;


int main(int argc, char *argv[])
{
    // loading mesh
    Eigen::MatrixXd V1;
    Eigen::MatrixXi F1;
    
	// Initialize Viewer
	igl::opengl::glfw::Viewer viewer;

	// Add the menu
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// New window
	menu.callback_draw_custom_window = [&]()
	{
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(320, 370), ImGuiSetCond_FirstUseEver);
		ImGui::Begin( "Tasks", nullptr, ImGuiWindowFlags_NoSavedSettings );
		
		if (ImGui::CollapsingHeader("Utilities", ImGuiTreeNodeFlags_DefaultOpen))
        {
        	// Task to run
			ImGui::SliderInt("Task", &viewerData.task, 0,7);
	        // Mesh to be used
	        ImGui::SliderInt("Mesh", &viewerData.mesh, 0,4);

	        if(viewerData.task == 1) {ImGui::Text("Task 1: Uniform Laplace (Mean)");}
			else if (viewerData.task == 2) {ImGui::Text("Task 1: Gaussian Curvature");}
			else if (viewerData.task == 3) {ImGui::Text("Task 3: Non-Uniform Laplace (Cotangent Mean)");}
	        else if (viewerData.task == 4) {ImGui::Text("Task 4: Spectral meshes");}
	        else if (viewerData.task == 5) {ImGui::Text("Task 5: Explicit Laplace Smoothing");}
			else if (viewerData.task == 6) {ImGui::Text("Task 6: Implicit Laplace Smoothing");}
			else if (viewerData.task == 7) {ImGui::Text("Task 7: Add Noise");}
	        else {ImGui::Text("Reset");}

	        if(viewerData.mesh == 0) {ImGui::Text("Mesh: lilium_s");}
	        else if (viewerData.mesh == 1) {ImGui::Text("Mesh: plane");}
	        else if (viewerData.mesh == 2) {ImGui::Text("Mesh: armadillo");}
	        else if (viewerData.mesh == 3) {ImGui::Text("Mesh: fandisk_ns");}
	        else{ImGui::Text("Mesh: plane_ns");}
	        ImGui::Text("");
        }

        if (ImGui::CollapsingHeader("Task 4", ImGuiTreeNodeFlags_DefaultOpen))
        {
        	ImGui::SliderInt("Num k Eigenvalues", &viewerData.k, 3, 100);
        }
    	if (ImGui::CollapsingHeader("Task 5/6", ImGuiTreeNodeFlags_DefaultOpen))
        {
        	ImGui::SliderInt("Iterations", &viewerData.numIters, 1, 500);
        	ImGui::InputFloat("Lambda", &viewerData.lambda, 0,0.001, "%.7f");
        }
    	if (ImGui::CollapsingHeader("Task 7", ImGuiTreeNodeFlags_DefaultOpen))
        {
        	ImGui::SliderFloat("STD Noise", &viewerData.Noise, 0, 1.0, "%.3f");
        }
        
		

        
        ImGui::Text("");
        if(ImGui::Button("Run")) {
        	std::cout << "Running..." << std::endl;
            viewerData.reset(viewer);
        }

        ImGui::End();
	};

	viewerData.reset(viewer);
	viewer.launch();
}

