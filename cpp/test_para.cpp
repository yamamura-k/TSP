#include <iostream>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include "SimulatedAnnealing.h"
#include "mpi.h"


template <class C>
void RootRecvVector(int src_rank, C &recv_buffer)
{
  int recvcount;
  MPI_Status st;
  MPI_Recv(&recvcount, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, &st);
  recv_buffer.resize(recvcount);
  MPI_Recv(&recv_buffer[0], recvcount * sizeof(recv_buffer[0]), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &st);
}
template <class C>
void SendVector(int dest_rank, C send_buffer)
{
  int sendcount = send_buffer.size();
  //MPI_Send(&sendcount, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
  //MPI_Send(&send_buffer[0], sendcount * sizeof(send_buffer[0]), MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD);
  
  MPI_Request req1, req2;
  MPI_Status st1, st2;
  MPI_Isend(&sendcount, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, &req1);
  MPI_Isend(&send_buffer[0], sendcount * sizeof(send_buffer[0]), MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD, &req2);
  MPI_Wait(&req1, &st1);
  MPI_Wait(&req2, &st2);
  
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
   std::vector<double> send_length_buffer;
   double ans = problem.solve(T, T_STOP, C);
   send_length_buffer.push_back(ans);

   SendVector(0, problem.best.tour);
   SendVector(0, send_length_buffer);
   //std::cout << "#" << rank << " Send solutions\n";

   if(rank == 0)
   {
      std::vector< std::vector<int> > recv_buffer;
      std::vector<double> recv_length_buffer;
      //const int dest_rank = 0;
      int src_rank;
      for(src_rank = 0; src_rank < size; src_rank++)
      {
         std::vector<int> incumbent;
         std::vector<double> incumbent_len;
         RootRecvVector(src_rank, incumbent);
         RootRecvVector(src_rank, incumbent_len);
         recv_buffer.push_back(incumbent);
         recv_length_buffer.push_back(incumbent_len[0]);
         //std::cout << "Receive solutions from " << src_rank << std::endl;
      }
      //std::vector<double>::iterator minIt = *std::min_element(recv_length_buffer.begin(), recv_length_buffer.end());
      auto minIt = std::min_element(recv_length_buffer.begin(), recv_length_buffer.end());
      size_t minIndex = std::distance(recv_length_buffer.begin(), minIt);
      std::cout << "rank: " << minIndex << "\ntour length: " << recv_length_buffer[minIndex] << "\ntour:\n";
      for(auto p : recv_buffer)
      {
         for(auto q : p) std::cout << q << " ";
      }
      std::cout << "\n";
      //for(auto l : recv_length_buffer) std::cout << "length = " << l << std::endl;
   }

   MPI_Finalize();
   return 0;
}