
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <omp.h>

#define MATRIX_SIZE_MAX 1000

void Cholesky_Decomposition(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, int& n)
{
    //omp_lock_t lock;

    //omp_init_lock(&lock);

    // Decomposing a matrix into Lower Triangular
#pragma omp parallel for num_threads(n)
    for (int i = 0; i < n; i++) 
    {
        //omp_set_lock(&lock);
        for (int j = 0; j <= i; j++) 
        {           
            double sum = 0;
            if (j == i) // summation for diagonals
            {
                for (int k = 0; k < j; k++)
                {
                    sum += pow(L[j][k], 2);
                } 

                L[j][j] = sqrt(A[j][j] - sum);
            }
            else 
            {
                // Evaluating L(i, j) using L(j, j)
                for (int k = 0; k < j; k++)
                {
                    sum += (L[i][k] * L[j][k]);                
                }   

                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
        //omp_unset_lock(&lock);
    }
    //omp_destroy_lock(&lock);

#pragma omp barrier
}

double Det(std::vector<std::vector<double>>& M, int n)
{
    double det = 0;
    //int submatrix[10][10];
    std::vector<std::vector<double>> submatrix(n, std::vector<double>(n, 0));
    if (n == 2)
        return ((M[0][0] * M[1][1]) - (M[1][0] * M[0][1]));
    else {
        for (int x = 0; x < n; x++) {
            int subi = 0;
            for (int i = 1; i < n; i++) {
                int subj = 0;
                for (int j = 0; j < n; j++) {
                    if (j == x)
                        continue;
                    submatrix[subi][subj] = M[i][j];
                    subj++;
                }
                subi++;
            }
            det = det + (pow(-1, x) * M[0][x] * Det(submatrix, n - 1));
        }
    }
    return det;
}

//Multiplying matrix a and b and storing in array mult
std::vector<std::vector<double>> Mult(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B, int& n)
{
    std::vector<std::vector<double> > M(n, std::vector<double>(n, 0));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                M[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return M;
}

std::vector<std::vector<double>> Minus(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B, int& n)
{
    std::vector<std::vector<double> > M(n, std::vector<double>(n, 0));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            M[i][j] = (A[i][j] - B[i][j]);
        }
    }

    return M;
}


int main()
{
    /*
    std::vector<std::vector<double>> vec
        {
        {7, 1, 3},
        {2, 4, 1},
        {1, 5, 1} 
        };
    std::cout << Det(vec, 3) << std::endl;

    */
    int n = 0;
    std::cout << "n : ";
    std::cin >> n;

    assert(n >= 3 && n < MATRIX_SIZE_MAX);

    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0));
    std::vector<std::vector<double> > L(n, std::vector<double>(n, 0));
    std::vector<std::vector<double> > U(n, std::vector<double>(n, 0));
    
    double Aij;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << "A[" << i + 1 << "][" << j + 1 << "] = ";
            std::cin >> Aij;
            A[i][j] = Aij;
        }
    }

    std::cout << std::endl << "Displaying A" << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << A[i][j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    Cholesky_Decomposition(A, L, n);


    std::cout << "Displaying Lower Triangular" << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << L[i][j] << "\t";
            U[j][i] = L[i][j];
        }
        
        std::cout << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Displaying Lower Triangular Transpose" << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << U[i][j] << "\t";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;

    auto Mul = Mult(L, U, n);
    auto Min = Minus(Mul, A, n);

    double X = Det(Min, n) / Det(A, n);
    std::cout << X;
    return 0;
}

