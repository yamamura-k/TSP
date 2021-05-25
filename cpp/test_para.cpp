#include <iostream>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include "SimulatedAnnealing.h"
#include "mpi.h"

template <class C>
void SendrecvVector(C &send_buffer, int dest_rank, C &recv_buffer, int src_rank)
{
  MPI_Status st;
  int sendcount = send_buffer.size();
  int recvcount;
  MPI_Sendrecv(&sendcount, 1, MPI_INT, dest_rank, 0, &recvcount, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, &st);
  recv_buffer.resize(recvcount);
  MPI_Sendrecv(&send_buffer[0], sendcount * sizeof(send_buffer[0]), MPI_BYTE, dest_rank, 0, &recv_buffer[0], recvcount * sizeof(recv_buffer[0]), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &st);
}

int main(int argc, char *argv[])
{
   std::cout <<"argc = " << argc << "\n" << argv[0] << "\n" << argv[1] << "\n";
   if(argc < 2)
   {
      std::cerr << "No Input file.\n";
      return 1;
   }
   int ncity;
   MatrixDouble D;
   Coord coord;
   double T = 1e10;
   double C = 0.995;
   double T_STOP = 1e-2;
   read(argv[1], ncity, D, coord);
   /*
   // for debug
   //今回は対称行列のみを扱うので、それをチェック
   for(int i = 0; i < ncity; i++)
   {
      for(int j = 0; j < ncity; j++)
      {
         std::cout << (D[i][j] == D[j][i]) << std::endl;
      }
   }
   */
   int size, rank;
   Tour best;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   SimulatedAnnealing problem(ncity, D);
   // なんかデッドロック的なことが起こってそうなのでまた後日デバッグする
   std::vector<int> recv_buffer;
   std::vector<double> send_length_buffer, recv_length_buffer;
   //const int dest_rank = 0;
   const int dest_rank = (rank-1+size) %size;
   const int src_rank = (rank-1+size) %size;
   double ans = problem.solve(T, T_STOP, C);
   send_length_buffer.push_back(ans);
   SendrecvVector(problem.best.tour, dest_rank, recv_buffer, src_rank);
   SendrecvVector(send_length_buffer, dest_rank, recv_length_buffer, src_rank);
   if(rank == 0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      //std::vector<double>::iterator minIt = *std::min_element(recv_length_buffer.begin(), recv_length_buffer.end());
      auto minIt = std::min_element(recv_length_buffer.begin(), recv_length_buffer.end());
      size_t minIndex = std::distance(recv_length_buffer.begin(), minIt);
      std::cout << "rank: " << minIndex << "\ntour length: " << recv_length_buffer[minIndex] << "\ntour:\n";
      for(auto p : recv_buffer)
      {
         std::cout << p << " ";
      }
      std::cout << "\n";
      for(double l : recv_length_buffer) std::cout << "length = " << l << std::endl;
   }

   MPI_Finalize();
   return 0;
}