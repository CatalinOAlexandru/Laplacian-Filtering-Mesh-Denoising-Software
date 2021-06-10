#include "laplace.h"
#include <random>
#include <igl/eigs.h>
#include <igl/bounding_box.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>


// H for mean curvature
Eigen::VectorXd getH(Eigen::MatrixXd V, Eigen::MatrixXi F, bool cotan){
    Eigen::SparseMatrix<double> L,C,invM;
    Eigen::VectorXd H;

    invM = invSparseMat(getM(V,F));

    if(cotan == false){
        C = uniformLaplace(V,F);
        L = 0.5*invM*C;
    } 
    else {
        C = nonUniLaplace(V,F);
        L = 0.5*invM*C;
    }
    
    H = 0.5*(L*V).rowwise().norm();
    return H;
}

// Getting the diagonal matrix M
Eigen::SparseMatrix<double> getM(Eigen::MatrixXd V, Eigen::MatrixXi F){
    Eigen::SparseMatrix<double> M;
    M.resize(V.rows(),V.rows());
    M.setZero();
    std::vector<std::vector<int> > VF;  // incident faces
    std::vector<std::vector<int> > VFi; // index of VF

    igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

    for (int vertex = 0; vertex<V.rows(); vertex++) {
        double totalAreaVertex = 0.0;
        // loop through all the faces (neigbours of vertex)
        for (int neighbors = 0; neighbors < VF[vertex].size(); neighbors++) {
            // current face and calculate the area
            Eigen::RowVector3i verticesNewFace = F.row(VF[vertex][neighbors]);
            Eigen::RowVector3d v1 = V.row(verticesNewFace(0));
            Eigen::RowVector3d v2 = V.row(verticesNewFace(1));
            Eigen::RowVector3d v3 = V.row(verticesNewFace(2));

            totalAreaVertex += getArea(v1, v2, v3);;
        }
        M.coeffRef(vertex, vertex) = totalAreaVertex / 3.0;
    }
    return M;
}

// Calculate the area of the triangle using 3 vertices
double getArea(Eigen::RowVector3d a, Eigen::RowVector3d b, Eigen::RowVector3d c) {
    Eigen::RowVector3d ab = a - b;
    Eigen::RowVector3d bc = b - c;
    double area = (ab.cross(bc)).norm() / 2.0;
    return area;
}


// Inverts a sparce matrix
Eigen::SparseMatrix<double> invSparseMat(Eigen::SparseMatrix<double> M){
    Eigen::SparseMatrix<double> invM;
    invM.resize(M.rows(),M.cols());
    invM.setZero();
    for (int i = 0; i <M.rows() ; i++) {
        invM.coeffRef(i,i) = 1/M.coeffRef(i,i);
    }
    return invM ;
}


// Tesk 1.1: Compute the uniform laplace operator uL which will have dimensions of |numVertices|x|numVertices| 
Eigen::SparseMatrix<double> uniformLaplace(Eigen::MatrixXd V, Eigen::MatrixXi F){
    Eigen::SparseMatrix<double> uL;
    uL.resize(V.rows(),V.rows());
    uL.setZero();
    std::vector<std::vector<int> > VF;  // incident faces
    std::vector<std::vector<int> > VFi; // index of VF
    igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

    // looping trough each vertex
    for (int vertex = 0; vertex<V.rows(); vertex++) {
        int k = VF[vertex].size();

        // loop through all the faces (neigbours of vertex)
        for (int neighbor = 0; neighbor < VF[vertex].size(); neighbor++) {
            Eigen::RowVector3i curFaceVertices = F.row(VF[vertex][neighbor]);

            // looping thorugh all the neighbors vertices
            for (int verticesNewFace = 0;
                // current face
                verticesNewFace < curFaceVertices.size(); verticesNewFace++) {
                int indexVert = curFaceVertices(verticesNewFace);
                double cf = -1/double(k);

                if(vertex==indexVert){ uL.coeffRef(vertex, indexVert) = 1;} 
                else{ uL.coeffRef(vertex, indexVert) = cf;}
            }
        }
    }
    return uL;
}


// Task 1.2: Extimate the gaussian curvature (gausC)
Eigen::VectorXd estimateGaussCurv(Eigen::MatrixXd V, Eigen::MatrixXi F){
    Eigen::VectorXd gausC;
    gausC.resize(V.rows());
    gausC.setZero();
    
    std::vector<std::vector<int> > VF;  // incident faces
    std::vector<std::vector<int> > VFi; // index of VF
    igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

    Eigen::SparseMatrix<double> M = getM(V,F);

    // looping trough each vertex
    for (int vertex = 0; vertex<V.rows(); vertex++) {
        double totalAngleVert = 0.0;
        // loop through all the faces (neigbours of vertex)
        for (int neighbor = 0; neighbor < VF[vertex].size(); neighbor++) {
            // current face
            Eigen::RowVector3i verticesNewFace = F.row(VF[vertex][neighbor]);
            // calculate area of the face
            Eigen::RowVector3d v1 = V.row(verticesNewFace(0));
            Eigen::RowVector3d v2 = V.row(verticesNewFace(1));
            Eigen::RowVector3d v3 = V.row(verticesNewFace(2));
    
            totalAngleVert += getAngle(v1, v2, v3);;           
        }   
        double pi = 3.14159265;
        gausC.coeffRef(vertex) = (2*pi - totalAngleVert)/M.coeffRef(vertex,vertex);
    }
    return gausC;
}



// Task 3: Get the laplace operator using cotangent (cL) discetization. Returns sparce matrox
Eigen::SparseMatrix<double> nonUniLaplace(Eigen::MatrixXd V, Eigen::MatrixXi F){
    
    Eigen::SparseMatrix<double> cL;
    cL.resize(V.rows(),V.rows());
    cL.setZero();
    std::vector<std::vector<int> > VF;  // incident faces
    std::vector<std::vector<int> > VFi; // index of VF
    igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

    // looping trough each vertex
    for (int vertex = 0; vertex<V.rows(); vertex++) {

        Eigen::RowVector3d p1 = V.row(vertex);
        Eigen::RowVector3d p2;
        Eigen::RowVector3d pt1;
        Eigen::RowVector3d pt2;

        // loop through all the faces (neigbours of vertex)
        for (int neighbor = 0; neighbor < VF[vertex].size(); neighbor++) {

            int curFace = VF[vertex][neighbor];
            int indexVertCurFace = VFi[vertex][neighbor]; // index for vertices in current face

            Eigen::RowVector3i curFaceVertices = F.row(curFace);
            int indexP2 = curFaceVertices((indexVertCurFace+1)%3);
            p2 = V.row(indexP2);
            pt1 = V.row(curFaceVertices((indexVertCurFace+2)%3));

            
            bool found = false;
            int j = 0;

            // search for pt2
            while(found==false && j<VF[vertex].size()){
                int nextFace = VF[vertex][j];
                if(nextFace == curFace){
                    j++;
                    continue;
                }

                Eigen::RowVector3i nextFaceVerts = F.row(nextFace);
                for (int i = 0; i < nextFaceVerts.size(); i++) {
                    if(nextFaceVerts(i) == indexP2){
                        if(nextFaceVerts((i+1)%3)!= vertex){ pt2 = V.row(nextFaceVerts((i+1)%3));} 
                        else{ pt2 = V.row(nextFaceVerts((i+2)%3));}
                        found = true;
                        break;
                    }
                }
                j++;
            }

            // Calculate cotangent discretization
            double alphaAngle = getAngle(pt1,p1,p2);
            double betaAngle = getAngle(pt2,p1,p2);
            double cIJ = cos(alphaAngle)/sin(alphaAngle) + cos(betaAngle)/sin(betaAngle);
            cL.coeffRef(vertex,indexP2) = cIJ;
            cL.coeffRef(indexP2,vertex) = cIJ;
        }

    }
    for (int i = 0; i < cL.rows(); i++) {
        cL.coeffRef(i,i) = - cL.row(i).sum();
    }
    return cL;
}


// get the angle in radians between AB and AC
double getAngle(Eigen::RowVector3d a, Eigen::RowVector3d b, Eigen::RowVector3d c){
    Eigen::RowVector3d ab = b - a;
    Eigen::RowVector3d ac = c - a;
    double angle = (ab.dot(ac))/(ab.norm()*ac.norm());
    angle = acos(angle);
    return angle;
}



// Task 4: Reconstruct a mesh using k eigenvectors of the laplace operator
Eigen::MatrixXd meshReconstruction(Eigen::MatrixXd V, Eigen::MatrixXi F, int k){
    Eigen::SparseMatrix<double> M = getM(V,F);
    Eigen::SparseMatrix<double> C = nonUniLaplace(V,F);
    C = (-0.5*C).eval();
    Eigen::MatrixXd eigVec;
    Eigen::VectorXd eigVal;

    if(k<11){igl::eigs(C, M, k, igl::EIGS_TYPE_SM, eigVec, eigVal);} 
    else {
        Eigen::SparseMatrix<double> invM = invSparseMat(M);
        Eigen::SparseMatrix<double> invMh = invM.cwiseSqrt();
        C = invMh*C*invMh;
        Spectra::SparseGenMatProd<double> op(C);
        Spectra::SymEigsSolver< double, Spectra::SMALLEST_MAGN, Spectra::SparseGenMatProd<double> > eigs(&op, k, 2*k);
        eigs.init();
        int nconv = eigs.compute();
        if(eigs.info() == Spectra::SUCCESSFUL){
            eigVal = eigs.eigenvalues();
            eigVec = eigs.eigenvectors().real();
            eigVec = invMh*eigVec;
        }
    }

    // igl::eigs(C, M, k, igl::EIGS_TYPE_SM, eigVec, eigVal);

    int numCols = eigVec.cols();

    Eigen::MatrixXd reconMesh;
    reconMesh.resize(V.rows(),V.cols());
    reconMesh.setZero();
    for (int freq = 0; freq < k; freq++) {
        Eigen::MatrixXd a;
        a = V.transpose() *M* eigVec.col(numCols-1-freq);
        reconMesh.col(0) += a(0)*eigVec.col(numCols-1-freq);
        reconMesh.col(1) += a(1)*eigVec.col(numCols-1-freq);
        reconMesh.col(2) += a(2)*eigVec.col(numCols-1-freq);
    }
    return reconMesh;
}



// Task 5: Smooth meshes using explicit laplace
Eigen::MatrixXd explicitSmoothing(Eigen::MatrixXd V, Eigen::MatrixXi F,double lambda, int numIters){
    Eigen::SparseMatrix<double> C,M,invM,L;
    C = nonUniLaplace(V,F);
    M = 2*getM(V,F);
    invM = invSparseMat(M);
    L = invM*C;

    Eigen::MatrixXd newMesh = V;
    double tol = 1e-6;

    for (int i = 0; i < numIters; i++) {
        Eigen::MatrixXd prev = newMesh;

        newMesh = newMesh + lambda*L*newMesh;
        std::cout << "Iter: "<< i+1 <<" | Norm: " << (newMesh-prev).norm() << std::endl;
    }
    return newMesh;
}


// Task 6: Smooth meshes using implicit laplace
Eigen::MatrixXd implicitSmoothing(Eigen::MatrixXd V, Eigen::MatrixXi F,double lambda, int numIters){

    Eigen::SparseMatrix<double> C,M,A;
    C = nonUniLaplace(V,F);
    M = 2*getM(V,F);
    A = M - lambda*C;

    Eigen::MatrixXd newMesh = V;
    double tol = 1e-6;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    for (int k = 0; k <numIters ; k++) {
        Eigen::MatrixXd prev = newMesh;

        newMesh = solver.solve(M*newMesh);
        std::cout << "Iter: "<< k+1 << " | Norm: " << (newMesh-prev).norm() << std::endl;
    }
    return newMesh;
}


// Task 7: Just adds gaussian noise to the mesh
Eigen::MatrixXd addNoise(Eigen::MatrixXd V, double std){
    Eigen::MatrixXd newMesh;
    newMesh.resize(V.rows(), V.cols());
    newMesh.setZero();

    Eigen::MatrixXd Vbound;
    Eigen::MatrixXi Fbound;

    igl::bounding_box(V, Vbound, Fbound);
    double noiseX = abs(Vbound.row(0).x()-Vbound.row(4).x());
    double noiseY = abs(Vbound.row(0).y()-Vbound.row(2).y());
    double noiseZ = abs(Vbound.row(0).z()-Vbound.row(1).z());

    // get random number
    std::default_random_engine rand;
    std::normal_distribution<double> gaussian(0.0, std);
    
    // Adding noise to each vertex
    for (int k=0; k<newMesh.rows(); k++){
        double x = gaussian(rand)*noiseX/100;
        double y = gaussian(rand)*noiseY/100;
        double z = gaussian(rand)*noiseZ/100;
        Eigen::RowVector3d gaussian_noise(x,y,z);
        newMesh.row(k) = V.row(k) + gaussian_noise;
    }
    return newMesh;
}