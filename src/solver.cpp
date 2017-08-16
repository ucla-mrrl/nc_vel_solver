#include<iostream>
#include <complex>
#include <vector>
#include <ctime>
#include<Eigen/Core>
#include<Eigen/LU>

#include "./include/cppoptlib/meta.h"
#include "./include/cppoptlib/problem.h"
#include "./include/cppoptlib/solver/neldermeadsolver.h"

using namespace std;
using namespace cppoptlib;
using namespace Eigen;


void vel_solve_weighted(int Ne, int n_voxels, complex<float> *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, float max_v, float *v_out, int grid);
void vel_solve(int Ne, int n_voxels, complex<float> *data_in, int *wraps_in, int *mask_in, float *A_in, float max_v, float *v_out);
void vel_solve_weighted_knownvel(int Ne, int n_voxels, complex<float> *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, float *v_in, float max_v, float *v_out);
void vel_solve_weighted_knownwraps(int Ne, int n_voxels, complex<float> *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, int *k_in, float max_v, float *v_out);

float cost(MatrixXf A, Vector3f v, VectorXcf data);
Vector3f solve_perm_vel(VectorXcf data, VectorXi wraps, MatrixXf A, MatrixXf Ap, float max_v);
Vector3f solve_perm_vel_weighted(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v);
Vector3f solve_NM_vel_weighted(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v, int grid);

VectorXf solve_NM_vel_weighted_reg(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v, int grid, float lam, MatrixXf v_neighbors, bool print_debug);

VectorXf solve_perm_vel_weighted_reg(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v, float lam, MatrixXf v_neighbors);
Vector3f solve_vel_weighted_knownwraps(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, VectorXi known_k);
Vector3f solve_vel_weighted_knownvel(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, VectorXf known_vel);

// Class for the cppoptlib library that gives the objective function with weighting but without regularization
class WeightedSolver : public Problem<float> {
    public:
        using typename cppoptlib::Problem<float>::Scalar;
        using typename cppoptlib::Problem<float>::TVector;

        MatrixXf A;
        MatrixXf w; 
        VectorXcf data;
        int Ne;
        

        float value(const TVector &v) {   
            const complex<float> ij(0, 1);             
            MatrixXcf diff(Ne, 1);
            MatrixXf Ave = A * v;
            for (int i = 0; i < Ne; i++) {
                diff(i) = w(i) * ( exp( ij * Ave(i) ) - data(i) );
            }
            float res = diff.norm();
            return res;
        }
};


// Class for cppoptlibrary that give objective function with weighting and Laplacian regularization
class WeightedSolverReg : public Problem<float> {
    public:
        using typename cppoptlib::Problem<float>::Scalar;
        using typename cppoptlib::Problem<float>::TVector;

        MatrixXf A;
        MatrixXf w; 
        VectorXcf data;
        int Ne;
        float lam; 
        MatrixXf v_neighbors;
        

        float value(const TVector &v) {   

            const complex<float> ij(0, 1);      

            MatrixXcf diff(Ne, 1);
            MatrixXf Ave = A * v;
            for (int i = 0; i < Ne; i++) {
                diff(i) = w(i) * ( exp( ij * Ave(i) ) - data(i) );
            }
            float data_reg = diff.norm();

            float vel_reg = 0;
            for (int i = 0; i < v_neighbors.rows(); i++) {
                Vector3f vs = v_neighbors.row(i);
                vel_reg += (v - vs).norm();
            }

            float res = data_reg + lam * vel_reg;

            return res;
        }
};

// N-dimensional subscript to linear index
int sub2ind(int *siz, int N, VectorXi sub)
{
	int idx =	0;
    for (int i = 0; i < N; i++)
    {
        int prod = 1;
        for (int j = N-1; j > i; j--)
            prod *= siz[j];
        idx += sub(i) * prod;
    }
	return idx;
}

// Linear index to N-dimensional subscripts
VectorXi ind2sub(int *siz, int N, int idx)
{
    int *prod = new int [N];
    VectorXi sub(N);
    for (int i = 0; i < N; i++)
    {
        prod[i] = 1;
        for (int j = N-1; j > i; j--)
            prod[i] *= siz[j];
    }
    
    for (int i = 0; i < N; i++)
    {
        sub(i) = idx ;
        for (int j = 0; j < i ; j++)
            sub(i) = sub(i) % prod[j];
        sub(i) = (int)floor( (float)sub(i) / prod[i] );
    }

    delete [] prod;

    return sub;
}

// This is the main solver for the complete weighted and regularized reconstruction
void reg_v2(int Ne, int n_dims, int *dims, int grid,
            int *mask, complex<float> *data_in, float *weights_in, int *wraps_in,
            float *A_in, float max_v, float lam, float *vel, float *vel_out,
            float *debug)
{
    Map<MatrixXf> Amap(A_in,3,Ne);
    MatrixXf A(Ne,3);
    A << Amap.transpose();

    Map<VectorXi> wraps(wraps_in, Ne);

    // Code for handling 2,3,4-D data while using the same loops
    Vector3i d0;
    Vector3i d1;
    Vector3i d2;
    Vector3i d3;
    int n_voxels0 = 0;

    if (n_dims == 2) {
        d0 << 0, 1, 0;
        d1 << 0, 1, 0;
        d2 << 0, dims[0], dims[0];
        d3 << 0, dims[1], dims[1];
        n_voxels0 = dims[0] * dims[1];
        lam /= (max_v * 4);
    } else if (n_dims == 3) {
        d0 << 0, 1, 0;
        d1 << 0, dims[0], dims[0];
        d2 << 0, dims[1], dims[1];
        d3 << 0, dims[2], dims[2];
        n_voxels0 = dims[0] * dims[1] * dims[2];
        lam /= (max_v * 6);
    } else if (n_dims == 4) {
        d0 << 0, dims[0], dims[0];
        d1 << 0, dims[1], dims[1];
        d2 << 0, dims[2], dims[2];
        d3 << 0, dims[3], dims[3];
        n_voxels0 = dims[0] * dims[1] * dims[2] * dims[3];
        lam /= (max_v * 8);
    }

    // Compute a list of all indices for easier looping later
    vector<int> all_ind(n_voxels0);

    int ii = 0;
    for (int i0 = d0[0]; i0 < d0[1]; i0++) {
    for (int i1 = d1[0]; i1 < d1[1]; i1++) {
    for (int i2 = d2[0]; i2 < d2[1]; i2++) {
    for (int i3 = d3[0]; i3 < d3[1]; i3++) {

        int mask_ind = i0 * d1[2] * d2[2] * d3[2] + i1 * d2[2] * d3[2] + i2 * d3[2] + i3;
        
        VectorXi sub(n_dims);
        if (n_dims==2) {sub << i2, i3;}
        if (n_dims==3) {sub << i1, i2, i3;}
        if (n_dims==4) {sub << i0, i1, i2, i3;}
        int mask_ind2 = sub2ind(dims, n_dims, sub);

        // if (mask_ind2 != mask_ind) {
        //     cout << "ERROR: ind matching: " << mask_ind2 << " " << mask_ind << endl;
        // }

        all_ind[ii] = mask_ind;

        VectorXi sub2 = ind2sub(dims, n_dims, mask_ind2);

        // cout << sub2 << endl;
        // cout << i0 << " " << i1 << " " << i2 << " " << i3 << endl << endl;

        ii++;

    }}}}

    // For in place regularization, approach the voxels randomly
    // Seems to work a little better/faster
    std::random_shuffle ( all_ind.begin(), all_ind.end() );

    #pragma omp parallel for
    for (int i0 = 0; i0 < all_ind.size(); i0++) {

        int mask_ind = all_ind[i0];

        if (mask[mask_ind] > 0) {
            
            // Matrix to hold all neighbor velocity vectors
            int good_neighbors = 0;
            if (n_dims == 2) {
                good_neighbors = 4;
            } else if (n_dims == 3) {
                good_neighbors = 6;
            } else if (n_dims == 4) {
                good_neighbors = 8;
            }
            MatrixXf v_neighbors(good_neighbors,3);

            VectorXi sub = ind2sub(dims, n_dims, mask_ind);
            int idx;
            int idx_lim;
            
            // Fill neighbor matrix
            int mod = 0;
            int v_ind = 0;
            if (n_dims > 0) {
                idx = sub(n_dims-1);
                idx_lim = dims[n_dims-1];
                mod = 1;
                
                if (idx < idx_lim-1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
                
                if (idx >= 1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
            }
            if (n_dims > 1) {
                idx = sub(n_dims-2);
                idx_lim = dims[n_dims-2];
                mod = d3[2];
                
                if (idx < idx_lim-1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
                
                if (idx >= 1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
            }
            if (n_dims > 2) {
                idx = sub(n_dims-3);
                idx_lim = dims[n_dims-3];
                mod = d3[2]*d2[2];

                if (idx < idx_lim-1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
                
                if (idx >= 1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
            }
            if (n_dims > 3) {
                idx = sub(n_dims-4);
                idx_lim = dims[n_dims-4];
                mod = d3[2]*d2[2]*d1[2];

                if (idx < idx_lim-1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
                
                if (idx >= 1) {
                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                } else {
                    v_neighbors.row(v_ind) << 0.0, 0.0, 0.0;
                }
                v_ind++;
            }

            Map<VectorXcf> data(&data_in[mask_ind*Ne], Ne);
            Map<VectorXf> weights(&weights_in[mask_ind*Ne], Ne);

            MatrixXf v_ref(v_neighbors.rows(),3);

            Vector3f ve;
            ve << vel[3*mask_ind], vel[3*mask_ind+1], vel[3*mask_ind+2];

            for (int i = 0; i < v_neighbors.rows(); i++) {
                v_ref.row(i) = ve;
            }

            float mag_norm = 0;
            for (int i = 0; i < v_neighbors.rows(); i++) {
                Vector3f vs = v_neighbors.row(i);
                mag_norm += (ve - vs).norm();
            }

            debug[6*mask_ind] = lam * mag_norm;
            // debug[6*mask_ind+1] = v_neighbors.row(0)[2];
            // debug[6*mask_ind+2] = v_neighbors.row(1)[2];
            // debug[6*mask_ind+3] = v_neighbors.row(2)[2];
            // debug[6*mask_ind+4] = v_neighbors.row(4)[2];

            bool print_debug = false;
            if ( (sub(0) == 5) && (sub(1) == 42) && (sub(2) == 9)) {print_debug = true;} 

            VectorXf v;
            if (grid > 0) {
                v = solve_NM_vel_weighted_reg(data, weights, wraps, A, max_v, grid, lam, v_neighbors, print_debug);
            } else {
                v = solve_perm_vel_weighted_reg(data, weights, wraps, A, max_v, lam, v_neighbors);
            }

            //In place change seems to help convergence, and results
            // i.e. voxels are unwrapped against voxels that have already been unwrapped in this iteration
            vel[3*mask_ind] = v[0];
            vel[3*mask_ind+1] = v[1];
            vel[3*mask_ind+2] = v[2];

            vel_out[3*mask_ind] = v[0];
            vel_out[3*mask_ind+1] = v[1];
            vel_out[3*mask_ind+2] = v[2];


            debug[6*mask_ind+2] = v[6];
            debug[6*mask_ind+3] = v[3];
            debug[6*mask_ind+4] = v[4];
            debug[6*mask_ind+5] = v[5];

        }
    }
}

//REMOVE THIS IN FINAl VERSION ITS JUST AN OLD VERISON
// TODO: Remove debugging array and change solver output back to 3
void reg_v1(int Ne, int n_dims, int *dims, int grid,
            int *mask, complex<float> *data_in, float *weights_in, int *wraps_in,
            float *A_in, float max_v, float lam, float *vel, float *vel_out,
            float *debug)
{
    Map<MatrixXf> Amap(A_in,3,Ne);
    MatrixXf A(Ne,3);
    A << Amap.transpose();

    Map<VectorXi> wraps(wraps_in, Ne);

    // Code for handling 2,3,4-D data while using the same loops
    Vector3i d0;
    Vector3i d1;
    Vector3i d2;
    Vector3i d3;
    int n_voxels0 = 0;

    if (n_dims == 2) {
        d0 << 0, 1, 0;
        d1 << 0, 1, 0;
        d2 << 1, dims[0]-1, dims[0];
        d3 << 1, dims[1]-1, dims[1];
        n_voxels0 = (dims[0]-2) * (dims[1]-2);
        lam /= (max_v * 4);
    } else if (n_dims == 3) {
        d0 << 0, 1, 0;
        d1 << 1, dims[0]-1, dims[0];
        d2 << 1, dims[1]-1, dims[1];
        d3 << 1, dims[2]-1, dims[2];
        n_voxels0 = (dims[0]-2) * (dims[1]-2) * (dims[2]-2);
        lam /= (max_v * 6);
    } else if (n_dims == 4) {
        d0 << 1, dims[0]-1, dims[0];
        d1 << 1, dims[1]-1, dims[1];
        d2 << 1, dims[2]-1, dims[2];
        d3 << 1, dims[3]-1, dims[3];
        n_voxels0 = (dims[0]-2) * (dims[1]-2) * (dims[2]-2) * (dims[3]-2);
        lam /= (max_v * 8);
    }

    // Compute a list of all indices for easier looping later
    vector<int> all_ind(n_voxels0);

    int ii = 0;
    for (int i0 = d0[0]; i0 < d0[1]; i0++) {
    for (int i1 = d1[0]; i1 < d1[1]; i1++) {
    for (int i2 = d2[0]; i2 < d2[1]; i2++) {
    for (int i3 = d3[0]; i3 < d3[1]; i3++) {

        int mask_ind = i0 * d1[2] * d2[2] * d3[2] + i1 * d2[2] * d3[2] + i2 * d3[2] + i3;
        all_ind[ii] = mask_ind;
        ii++;

    }}}}

    // For in place regularization, approach the voxels randomly
    // Seems to work a little better/faster
    std::random_shuffle ( all_ind.begin(), all_ind.end() );

    #pragma omp parallel for
    for (int i0 = 0; i0 < all_ind.size(); i0++) {

        int mask_ind = all_ind[i0];

        if (mask[mask_ind] > 0) {
            
            // Matrix to hold all neighbor velocity vectors
            int good_neighbors = 0;
            if (n_dims == 2) {
                good_neighbors = 4;
            } else if (n_dims == 3) {
                good_neighbors = 6;
            } else if (n_dims == 4) {
                good_neighbors = 8;
            }
            MatrixXf v_neighbors(good_neighbors,3);

            // Fill neighbor matrix
            int mod = 0;
            int v_ind = 0;
            if (n_dims > 0) {
                mod = 1;
                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                    v_ind++;
                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                    v_ind++;
            }
            if (n_dims > 1) {
                mod = d3[2];
                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                    v_ind++;
                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                    v_ind++;
            }
            if (n_dims > 2) {
                mod = d3[2]*d2[2];

                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                    v_ind++;

                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                    v_ind++;
            }
            if (n_dims > 3) {
                mod = d3[2]*d2[2]*d1[2];

                    v_neighbors.row(v_ind) << vel[3*(mask_ind + mod)], vel[3*(mask_ind + mod)+1], vel[3*(mask_ind + mod)+2];
                    v_ind++;

                    v_neighbors.row(v_ind) << vel[3*(mask_ind - mod)], vel[3*(mask_ind - mod)+1], vel[3*(mask_ind - mod)+2];
                    v_ind++;
            }

            Map<VectorXcf> data(&data_in[mask_ind*Ne], Ne);
            Map<VectorXf> weights(&weights_in[mask_ind*Ne], Ne);

            MatrixXf v_ref(v_neighbors.rows(),3);

            Vector3f ve;
            ve << vel[3*mask_ind], vel[3*mask_ind+1], vel[3*mask_ind+2];

            for (int i = 0; i < v_neighbors.rows(); i++) {
                v_ref.row(i) = ve;
            }

            float mag_norm = 0;
            for (int i = 0; i < v_neighbors.rows(); i++) {
                Vector3f vs = v_neighbors.row(i);
                mag_norm += (ve - vs).norm();
            }

            debug[6*mask_ind] = lam * mag_norm;
            // debug[6*mask_ind+1] = v_neighbors.row(0)[2];
            // debug[6*mask_ind+2] = v_neighbors.row(1)[2];
            // debug[6*mask_ind+3] = v_neighbors.row(2)[2];
            // debug[6*mask_ind+4] = v_neighbors.row(4)[2];

            VectorXf v;
            if (grid > 0) {
                v = solve_NM_vel_weighted_reg(data, weights, wraps, A, max_v, grid, lam, v_neighbors, false);
            } else {
                v = solve_perm_vel_weighted_reg(data, weights, wraps, A, max_v, lam, v_neighbors);
            }

            //In place change seems to help convergence, and results
            // i.e. voxels are unwrapped against voxels that have already been unwrapped in this iteration
            vel[3*mask_ind] = v[0];
            vel[3*mask_ind+1] = v[1];
            vel[3*mask_ind+2] = v[2];

            vel_out[3*mask_ind] = v[0];
            vel_out[3*mask_ind+1] = v[1];
            vel_out[3*mask_ind+2] = v[2];


            debug[6*mask_ind+2] = v[6];
            debug[6*mask_ind+3] = v[3];
            debug[6*mask_ind+4] = v[4];
            debug[6*mask_ind+5] = v[5];

        }
    }
}

// Solve velocity with weighting given a known velocity.  The known velocity is used solely for generating an unwrapping vector
// So this is the function that removes the regularization smoothing effects of reg_v2()
void vel_solve_weighted_knownvel(int Ne, int n_voxels, complex<float> *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, float *v_in, float max_v, float *v_out)
{
    Map<MatrixXf> Amap(A_in,3,Ne);
    MatrixXf A(Ne,3);
    A << Amap.transpose();

    Map<VectorXi> wraps(wraps_in, Ne);

    #pragma omp parallel for
    for(int i = 0; i < n_voxels; i++)
    {
        if (mask_in[i] > 0) {
            Map<VectorXcf> data(&data_in[i*Ne], Ne);

            Map<VectorXf> weights(&weights_in[i*Ne], Ne);
            Map<VectorXf> kv(&v_in[i*3], 3);

            VectorXf v = solve_vel_weighted_knownvel(data, weights, wraps, A, kv);

            v_out[i*3] = v[0];
            v_out[i*3+1] = v[1];
            v_out[i*3+2] = v[2];
        } else {
            v_out[i*3] = 0;
            v_out[i*3+1] = 0;
            v_out[i*3+2] = 0;
        }
    }
}

// Similar to preious function but with unwrapping vecotr as input, I believe this is not currently used by anything.
void vel_solve_weighted_knownwraps(int Ne, int n_voxels, complex<float> *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, int *k_in, float max_v, float *v_out)
{
    Map<MatrixXf> Amap(A_in,3,Ne);
    MatrixXf A(Ne,3);
    A << Amap.transpose();

    Map<VectorXi> wraps(wraps_in, Ne);

    #pragma omp parallel for
    for(int i = 0; i < n_voxels; i++)
    {
        if (mask_in[i] > 0) {
            Map<VectorXcf> data(&data_in[i*Ne], Ne);

            Map<VectorXf> weights(&weights_in[i*Ne], Ne);
            Map<VectorXi> kk(&k_in[i*Ne], Ne);

            VectorXf v = solve_vel_weighted_knownwraps(data, weights, wraps, A, kk);

            v_out[i*3] = v[0];
            v_out[i*3+1] = v[1];
            v_out[i*3+2] = v[2];
        } else {
            v_out[i*3] = 0;
            v_out[i*3+1] = 0;
            v_out[i*3+2] = 0;
        }
    }
}

// Nonconvex solver with weighting but no regularization, either with multi-start Nelder-Mead or permutation checking (Zwart method)
void vel_solve_weighted(int Ne, int n_voxels, complex<float> *data_in, float *weights_in, int *wraps_in, int *mask_in, float *A_in, float max_v, float *v_out, int grid)
{
    Map<MatrixXf> Amap(A_in,3,Ne);
    MatrixXf A(Ne,3);
    A << Amap.transpose();

    Map<VectorXi> wraps(wraps_in, Ne);

    #pragma omp parallel for
    for(int i = 0; i < n_voxels; i++)
    {
        if (mask_in[i] > 0) {
            Map<VectorXcf> data(&data_in[i*Ne], Ne);

            Map<VectorXf> weights(&weights_in[i*Ne], Ne);

            VectorXf v;
            if (grid > 0) {
                v = solve_NM_vel_weighted(data, weights, wraps, A, max_v, grid);
            } else {
                v = solve_perm_vel_weighted(data, weights, wraps, A, max_v);
            }

            v_out[i*3] = v[0];
            v_out[i*3+1] = v[1];
            v_out[i*3+2] = v[2];
        } else {
            v_out[i*3] = 0;
            v_out[i*3+1] = 0;
            v_out[i*3+2] = 0;
        }
    }
}

// Zwart solver of MDHM data
void vel_solve(int Ne, int n_voxels, complex<float> *data_in, int *wraps_in, int *mask_in, float *A_in, float max_v, float *v_out)
{
    Map<MatrixXf> Amap(A_in,3,Ne);
    MatrixXf A(Ne,3);
    A << Amap.transpose();

    MatrixXf Ap = (A.transpose()*A).inverse() * A.transpose();

    Map<VectorXi> wraps(wraps_in, Ne);

    #pragma omp parallel for
    for(int i = 0; i < n_voxels; i++)
    {
        if (mask_in[i] > 0) {
            Map<VectorXcf> data(&data_in[i*Ne], Ne);
            VectorXf v = solve_perm_vel(data, wraps, A, Ap, max_v);

            v_out[i*3] = v[0];
            v_out[i*3+1] = v[1];
            v_out[i*3+2] = v[2];
        } else {
            v_out[i*3] = 0;
            v_out[i*3+1] = 0;
            v_out[i*3+2] = 0;
        }
    }

}

// Cost functions that are used with the Zwart-type permutation solvers
float cost(MatrixXf A, Vector3f v, VectorXcf data)
{
    int Ne = data.size();
    const complex<float> ij(0, 1);
    MatrixXcf diff(Ne, 1);

    MatrixXf Ave = A * v;

    for (int i = 0; i < Ne; i++) {
        diff(i) = ( exp( ij * Ave(i) ) - data(i) );
    }

    return diff.norm();
}

// Cost functions that are used with the Zwart-type permutation solvers
float cost_weighted(MatrixXf A, MatrixXf w, Vector3f v, VectorXcf data)
{
    int Ne = data.size();
    const complex<float> ij(0, 1);
    MatrixXcf diff(Ne, 1);

    MatrixXf Ave = A * v;

    for (int i = 0; i < Ne; i++) {
        diff(i) = w(i) * ( exp( ij * Ave(i) ) - data(i) );
    }

    float res = diff.norm();

    return res;
}

// This is the primary solver (weighted + regularized) for a single voxel.  Uses a multi-start Nelder-Mead approach
VectorXf solve_NM_vel_weighted_reg(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v, int grid, float lam, MatrixXf v_neighbors, bool print_debug)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    VectorXf ve(3);
    MatrixXf phi_uw(Ne, 1);

    float min_res = 1e30;
    Matrix<float, 3, 1> min_v;
    min_v << 0.0, 0.0, 0.0;

    WeightedSolverReg f;
    f.A = A;
    f.w = weights;
    f.data = data;
    f.Ne = Ne;
    f.lam = lam;
    f.v_neighbors = v_neighbors;
    // GradientDescentSolver<WeightedSolverReg> solver;
    
    NelderMeadSolver<WeightedSolverReg> solver;
    Criteria<float> crit = Criteria<float>::defaults();
    crit.xDelta = max_v/100.0;
    // crit.iterations = 50;
    solver.setStopCriteria(crit);

    // // NewtonDescentSolver<WeightedSolverReg> solver;
    // // ConjugatedGradientDescentSolver<WeightedSolverReg> solver;
    // BfgsSolver<WeightedSolverReg> solver;
    // // GradientDescentSolver<WeightedSolverReg> solver;
    // Criteria<float> crit = Criteria<float>::defaults();
    // crit.gradNorm = 0.01;  
    // // crit.iterations = 40; 
    // solver.setStopCriteria(crit);

    if (print_debug) {cout << "Stop criteria values: " << endl << crit << endl;}

    int debug_count = 0;

    float d_v = 2.0 * max_v / (float)grid;
    for(float vz = -max_v + d_v/2.0; vz <= max_v; vz += d_v)
    for(float vy = -max_v + d_v/2.0; vy <= max_v; vy += d_v)
    for(float vx = -max_v + d_v/2.0; vx <= max_v; vx += d_v)
    {{{
        ve << vx, vy, vz;
        if (ve.norm() <= max_v) {
 
            solver.minimize(f, ve);
            float res = f(ve);
            
            if ( (res < min_res) && (ve.norm() <= max_v) ) {
                min_res = res;
                min_v = ve;
            }

            if ((ve.norm() > 1.0) && (ve.norm() < max_v) && print_debug && (debug_count < 3)) {
                debug_count++;
                cout << "Solver status: " << solver.status() << endl;
                cout << "Final criteria values: " << endl << solver.criteria() << endl;
            }
        }

    }}}

    return min_v;
}

// This solves the weighted and regularized objective function using the Zwart permutation method
VectorXf solve_perm_vel_weighted_reg(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v, float lam, MatrixXf v_neighbors)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    Matrix<float, 3, 1> ve;
    MatrixXf phi_uw(Ne, 1);

    // Scale weights so regularization is even
    float weight_norm = weights.norm()/Ne;
    for(int i = 0; i < Ne; i++) {
        weights(i) /= weight_norm;
    }

    MatrixXf W = MatrixXf::Zero(Ne, Ne);;
    W.diagonal() = weights;
    MatrixXf WA = W * A;
    MatrixXf WAp = (WA.transpose()*WA).inverse() * WA.transpose();

    float min_res = 1e30;
    Matrix<float, 3, 1> min_v;
    min_v << 0.0, 0.0, 0.0;
    
    float min_vel_reg = 0;
    float min_data_reg = 0;

    // Number of wrapping permutations
    int num_perm = 1;
    for(int i = 0; i < Ne; i++) {
        num_perm *= (wraps(i)*2 + 1);
    }

    // Initialize first wrapping test
    MatrixXf k(Ne, 1);
    for(int i = 0; i < Ne; i++) {
        k(i) = -wraps(i);
    }

    // Check all wrapping permutations for best velocity
    for(int j = 0; j < num_perm; j++) {

        for (int i = 0; i < Ne; i++) {
            phi_uw(i) = arg(data(i)) + 2.0 * pi * k(i);
        }
        ve = WAp * (W * phi_uw);

        float vel_reg = 0;
        for (int i = 0; i < v_neighbors.rows(); i++) {
            Vector3f vs = v_neighbors.row(i);
            vel_reg += (ve - vs).norm();
        }

        float data_reg = cost_weighted(A, weights, ve, data);

        float res = data_reg + lam * vel_reg;

        if ( (res < min_res) && (ve.norm() <= max_v) ) {
            min_res = res;
            min_v = ve;
            min_data_reg = data_reg;
            min_vel_reg = vel_reg * lam;
        }

        // Set the next permutation of k
        for(int i = 0; i < Ne; i++) {
            if (k(i) < wraps(i)) {
                k(i) += 1;
                break;
            } else {
                k(i) = -wraps(i);
            }
        }
    }

    Matrix<float, 9, 1> v_out;
    v_out << min_v[0], min_v[1], min_v[2], min_res, min_data_reg, min_vel_reg, weight_norm;
    return v_out;
}

// ---------
// Below are all single voxel solvers using different combinations of:
// - weighted or non-weighted
// - full nonconvex methods or permutation checking
// ---------


Vector3f solve_perm_vel(VectorXcf data, VectorXi wraps, MatrixXf A, MatrixXf Ap, float max_v)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    Matrix<float, 3, 1> ve;
    MatrixXf phi_uw(Ne, 1);

    float min_res = 1e30;
    Matrix<float, 3, 1> min_v;
    min_v << 0.0, 0.0, 0.0;

    // Number of wrapping permutations
    int num_perm = 1;
    for(int i = 0; i < Ne; i++) {
        num_perm *= (wraps(i)*2 + 1);
    }

    // Initialize first wrapping vector
    MatrixXf k(Ne, 1);
    for(int i = 0; i < Ne; i++) {
        k(i) = -wraps(i);
    }

    // Check all wrapping permutations for best velocity
    for(int j = 0; j < num_perm; j++) {

        for (int i = 0; i < Ne; i++) {
            phi_uw(i) = arg(data(i)) + 2.0 * pi * k(i);
        }
        ve = Ap * phi_uw;

        float res = cost(A, ve, data);

        if ( (res < min_res) && (ve.norm() <= max_v) ) {
            min_res = res;
            min_v = ve;
        }

        // Set the next permutation of k
        for(int i = 0; i < Ne; i++) {
            if (k(i) < wraps(i)) {
                k(i) += 1;
                break;
            } else {
                k(i) = -wraps(i);
            }
        }

    }

    return min_v;
}



Vector3f solve_vel_weighted_knownvel(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, VectorXf known_vel)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    Matrix<float, 3, 1> ve;
    MatrixXf phi_uw(Ne, 1);

    MatrixXf W = MatrixXf::Zero(Ne, Ne);;
    W.diagonal() = weights;
    MatrixXf WA = W * A;
    MatrixXf WAp = (WA.transpose()*WA).inverse() * WA.transpose();

    MatrixXf Akv = A * known_vel;

    float k;
    for (int i = 0; i < Ne; i++) {
        k = round( ( Akv(i) - arg(data(i)) ) / (2.0 * pi) );
        phi_uw(i) = arg(data(i)) + 2.0 * pi * k;
    }

    ve = WAp * (W * phi_uw);

    return ve;
}


Vector3f solve_vel_weighted_knownwraps(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, VectorXi known_k)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    Matrix<float, 3, 1> ve;
    MatrixXf phi_uw(Ne, 1);

    MatrixXf W = MatrixXf::Zero(Ne, Ne);;
    W.diagonal() = weights;
    MatrixXf WA = W * A;
    MatrixXf WAp = (WA.transpose()*WA).inverse() * WA.transpose();

    for (int i = 0; i < Ne; i++) {
        phi_uw(i) = arg(data(i)) + 2.0 * pi * known_k(i);
    }

    ve = WAp * (W * phi_uw);

    return ve;
}


Vector3f solve_NM_vel_weighted(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v, int grid)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    VectorXf ve(3);
    MatrixXf phi_uw(Ne, 1);

    float min_res = 1e30;
    Matrix<float, 3, 1> min_v;
    min_v << 0.0, 0.0, 0.0;

    WeightedSolver f;
    f.A = A;
    f.w = weights;
    f.data = data;
    f.Ne = Ne;
    NelderMeadSolver<WeightedSolver> solver;

    Criteria<float> crit = Criteria<float>::defaults();
    // cout << "Default criteria values: " << endl << crit << endl;
    crit.xDelta = max_v/100.0;
    crit.iterations = 50;
    solver.setStopCriteria(crit);

    float d_v = 2.0 * max_v / (float)grid;
    for(float vz = -max_v + d_v/2.0; vz <= max_v; vz += d_v)
    for(float vy = -max_v + d_v/2.0; vy <= max_v; vy += d_v)
    for(float vx = -max_v + d_v/2.0; vx <= max_v; vx += d_v)
    {{{
        ve << vx, vy, vz;
        if (ve.norm() <= max_v) {
            // cout << "Start criteria values: " << endl << solver.criteria() << endl;
            solver.minimize(f, ve);
            float res = f(ve);
            // cout << "Solver status: " << solver.status() << endl;
            // cout << "Final criteria values: " << endl << solver.criteria() << endl;
            if ( (res < min_res) && (ve.norm() <= max_v) ) {
                min_res = res;
                min_v = ve;
            }
        }

    }}}

    return min_v;
}



Vector3f solve_perm_vel_weighted(VectorXcf data, VectorXf weights, VectorXi wraps, MatrixXf A, float max_v)
{
    const float pi = acos(-1);
    const complex<float> ij(0, 1);

    int Ne = data.size();

    Matrix<float, 3, 1> ve;
    MatrixXf phi_uw(Ne, 1);

    MatrixXf W = MatrixXf::Zero(Ne, Ne);;
    W.diagonal() = weights;
    MatrixXf WA = W * A;
    MatrixXf WAp = (WA.transpose()*WA).inverse() * WA.transpose();

    float min_res = 1e30;
    Matrix<float, 3, 1> min_v;
    min_v << 0.0, 0.0, 0.0;

    // Number of wrapping permutations
    int num_perm = 1;
    for(int i = 0; i < Ne; i++) {
        num_perm *= (wraps(i)*2 + 1);
    }

    // Initialize first wrapping test
    MatrixXf k(Ne, 1);
    for(int i = 0; i < Ne; i++) {
        k(i) = -wraps(i);
    }

    // Check all wrapping permutations for best velocity
    for(int j = 0; j < num_perm; j++) {

        for (int i = 0; i < Ne; i++) {
            phi_uw(i) = arg(data(i)) + 2.0 * pi * k(i);
        }
        ve = WAp * (W * phi_uw);

        float res = cost_weighted(A, weights, ve, data);

        if ( (res < min_res) && (ve.norm() <= max_v) ) {
            min_res = res;
            min_v = ve;
        }

        // Set the next permutation of k
        for(int i = 0; i < Ne; i++) {
            if (k(i) < wraps(i)) {
                k(i) += 1;
                break;
            } else {
                k(i) = -wraps(i);
            }
        }
    }

    return min_v;
}