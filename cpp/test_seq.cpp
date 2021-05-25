#include <iostream>
#include "SimulatedAnnealing.h"
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
   SimulatedAnnealing problem(ncity, D);
   double ans = problem.solve(T, T_STOP, C);
   std::cout << "[ Best incumbent solution ] " << std::endl;
   problem.dispTour(problem.best);
   std::cout << problem.calcTourLen(problem.best) << std::endl;
   return 0;
}