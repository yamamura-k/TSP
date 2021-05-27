#include "SimulatedAnnealing.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <cassert>
#include <limits>

typedef std::vector< std::vector<double> > MatrixDouble;
typedef std::vector<double> VectorDouble;
typedef std::map<int, Coordinate> Coord;
typedef double (*FUNCTYPE)(Coordinate A, Coordinate B);

double calcDistGeo(Coordinate A, Coordinate B)
{
   double laRe, loRe, NSD, EWD, distance;
   laRe = deg2rad(B.y - A.y);
   loRe = deg2rad(B.x - A.x);
   NSD = earth_r*laRe;
   EWD = std::cos(deg2rad(A.y))*earth_r*loRe;
   distance = std::sqrt(std::pow(NSD,2)+std::pow(EWD,2));
   return distance;
}
double calcDistEUC2D(Coordinate A, Coordinate B)
{
   double distance;
   distance = std::sqrt(std::pow((A.x-B.x),2)+std::pow((A.y-B.y),2));
   return distance;
}
int read(std::string filename, int& ncity, MatrixDouble& D, Coord& coord)
{
   std::ifstream ifs(filename, std::ios::in);
   std::cout << "Function [ read ] called\n";
   if(!ifs)
   {
      std::cerr << "Error: cannot open file." << std::endl;
      return 1;
   }
   std::cout << "Successfully opened the input file\n";
   std::string tmp, name;
   std::string value;
   FUNCTYPE distFunc;
   std::cout << "Start reading\n";
   int line = 0;
   while(getline(ifs, tmp))
   {
      std::cout << "reading line at " << ++line << "\n";
      std::stringstream ss;
      ss << tmp;
      ss >> name >> value;
      if(name == "EOF") break;
      if(name == "DIMENSION:")
      {
         ncity = stoi(value);
      }
      else if(name == "EDGE_WEIGHT_TYPE:")
      {
         if(value == "GEO")
         {
            distFunc = calcDistGeo;
         }
         else if(value == "EUC_2D")
         {
            std::cout << value << std::endl;
            distFunc = calcDistEUC2D;
         }
         else
         {
            std::cerr << "Error: not supported" << std::endl;
            return 1;
         }
      }
      else if(name == "EDGE_WEIGHT_TYPE")
      {
         ss >> value;
         if(value == "GEO")
         {
            distFunc = calcDistGeo;
         }
         else if(value == "EUC_2D")
         {
            std::cout << value << std::endl;
            distFunc = calcDistEUC2D;
         }
         else
         {
            std::cerr << "Error: not supported" << std::endl;
            return 1;
         }
      }
      else if(name == "DISPLAY_DATA_TYPE:")
      {
         if(value != "COORD_DISPLAY")
         {
            std::cerr << "Error: not supported" << std::endl;
            return 1;
         }
      }
      else if(name == "NODE_COORD_SECTION")
      {
         int ind = -1;
         double x, y;
         for(int i = 0; i < ncity; i++)
         {
            ifs >> ind >> y >> x;
            coord[ind].x = x;
            coord[ind].y = y;
         }
      }
   }
   /*
   // for debug
   std::cout << "Check coordinates\n";
   for(int i = 0; i < ncity; ++i)
   {
      std::cout << i << ": x, y = " << coord[i].x << ", " << coord[i].y << std::endl;
   }
   */

   std::cout << "Initialize distance matrix\n";
   D.resize(ncity, VectorDouble(ncity, 0));
   std::cout << "Start caluculating distance matrix\n";
   std::cout << D.size() << ", " << D[0].size() << std::endl;
   for(int i = 0; i < ncity; i++)
   {
      for(int j = i; j < ncity; j++)
      {
         D[i][j] = distFunc(coord[i], coord[j]);
         D[j][i] = D[i][j];
         //D[j][i] = distFunc(coord[j], coord[i]);
      }
   }
   std::cout << "Finish reading the input file\n";
   return 0;
};
SimulatedAnnealing::SimulatedAnnealing(int _ncity, MatrixDouble _D)
{
   ncity = _ncity;
   D = _D;
   current.tour.resize(ncity);
   best.tour.resize(ncity);
   std::mt19937 tmp(seed_gen());
   engine = tmp;
   std::uniform_real_distribution<> uni_dst(0, 1);
   rand = uni_dst;
   std::uniform_int_distribution<> nxt(0, ncity-1);
   next_target = nxt;
   current = initialize_random();
   std::copy(current.tour.begin(), current.tour.end(), best.tour.begin());
   best.length = current.length;
}
Tour SimulatedAnnealing::initialize_random()
{
   Tour tmp;
   tmp.tour.resize(ncity);
   for(int i = 0; i < ncity; i++)
   {
      tmp.tour[i] = i;
   }
   std::shuffle(tmp.tour.begin(), tmp.tour.end(), engine);
   tmp.length = calcTourLen(tmp);
   //for debug
   //std::cout << "[ Initial Solution ]\n";
   //dispTour(tmp);
   return tmp;
}
void SimulatedAnnealing::dispTour(Tour tmp)
{
   std::cout << "total length: " << tmp.length << std::endl;
   for(auto p: tmp.tour)
   {
      std::cout << p << " ";
   }
   std::cout << "\n";
}
double SimulatedAnnealing::calcTourLen(Tour tmp)
{
   double len = D[tmp.tour[ncity-1]][tmp.tour[0]];
   for(int i = 0; i < ncity - 1; i++)
   {
      len += D[tmp.tour[i]][tmp.tour[i+1]];
   }
   return len;
}
bool SimulatedAnnealing::is_bug()
{
   double tour_len = calcTourLen(current);
   if(std::abs(current.length - tour_len) > EPS)
   {
      std::cout << "\nTour length is not correct!\n";
      std::cout << "current.length = " << current.length << ", " << "actual = " << tour_len << std::endl;
      return true;
   }
   for(int i = 0; i < ncity-1; ++i)
   {
      for(int j = i+1; j < ncity; ++j)
      {
         if(current.tour[i] == current.tour[j])
         {
            std::cout << "Bug is detected at index (i, j) = (" << i << ", " << j << ")" << std::endl; 
            return true;
         }
      }
   }
   return false;
}
double SimulatedAnnealing::calcSwapCost(int i, int j)
{
   int i_now, i_prev, i_nxt;
   int j_now, j_prev, j_nxt;
   i_now = current.tour[i];
   i_prev = current.tour[(ncity+i-1)%ncity];
   i_nxt = current.tour[(i+1)%ncity];
   j_now = current.tour[j];
   j_prev = current.tour[(ncity+j-1)%ncity];
   j_nxt = current.tour[(j+1)%ncity];
   /*
   // for debug
   std::cout << i_prev << ", "
             << i_now << ", "
             << i_nxt << " ("
             << (ncity+i-1)%ncity << ", "
             << i << ", "
             << (i+1)%ncity << ")\n";
   std::cout << j_prev << ", "
             << j_now << ", "
             << j_nxt << " ("
             << (ncity+j-1)%ncity << ", "
             << j << ", "
             << (j+1)%ncity << ")\n";
   std::cout << "TRY: calc current cost"<< std::endl;
   std::cout << "target indexes: " << (ncity+i-1)%ncity << ", " << i << ", " << (i+1)%ncity << std::endl;
   */
   double current_cost = D[i_prev][i_now] + D[i_now][i_nxt] + D[j_prev][j_now] + D[j_now][j_nxt];
   /*
   // for debug
   double tmp1 = D[i_now][i_nxt] + D[j_prev][j_now];
   if(std::abs(i - j) == 1) std::cout << tmp1
   << " : " << i_now
   << ", " << i_nxt
   << ", " << j_prev 
   << ", " << j_now 
   << ", " << D[i_now][i_nxt]
   << ", " << D[j_prev][j_now]
   << " : " << j_now
   << ", " << j_nxt
   << ", " << i_prev 
   << ", " << i_now 
   << ", " << D[j_now][j_nxt]
   << ", " << D[i_prev][i_now] << std::endl;
   std::cout << "FINISH: calc current cost"<< std::endl;
   */
   if(j_now == i_nxt)//隣接している場合の例外
   {
      i_nxt = i_now;
      j_prev = j_now;
   }
   if(i_now == j_nxt)//隣接している場合の例外
   {
      j_nxt = j_now;
      i_prev = i_now;
   }
   double new_cost = D[i_prev][j_now] + D[j_now][i_nxt] + D[j_prev][i_now] + D[i_now][j_nxt];
   /*
   // for debug
   double tmp2 = D[j_now][i_nxt] + D[j_prev][i_now];
   if(std::abs(i - j) == 1) std::cout << tmp2
   << " : " << j_now
   << ", " << i_nxt
   << ", " << j_prev 
   << ", " << i_now
   << ", " << D[j_now][i_nxt]
   << ", " << D[j_prev][i_now]
   << " : " << i_now
   << ", " << j_nxt
   << ", " << i_prev 
   << ", " << j_now
   << ", " << D[i_now][j_nxt]
   << ", " << D[i_prev][j_now] << std::endl;
   */
   Tour tmp;
   tmp.length = 0;
   tmp.tour = current.tour;
   std::swap(tmp.tour[i], tmp.tour[j]);
   if(std::abs(new_cost - current_cost - (calcTourLen(tmp)-calcTourLen(current)))>EPS)
   {
      std::cout << std::endl;
      std::cout << "Abnormal behaviour\n";
      if(std::abs(i-j) == 1)
      {
         std::cout << i_prev << ", "
                   << j_now << ", "
                   << i_nxt << "\n";
         std::cout << j_prev << ", "
                   << i_now << ", "
                   << j_nxt << "\n";
      }
      dispTour(current);
      dispTour(tmp);
      if(std::abs(i-j) == 1)
      {
         std::cout << D[j_now][i_nxt] - D[j_prev][i_now] << std::endl;
         std::cout << D[j_now][i_nxt] - D[i_nxt][j_now] << std::endl;
      }
      std::cout << new_cost - current_cost << std::endl;
      std::cout << calcTourLen(tmp)-calcTourLen(current) << std::endl;
   }
   return new_cost - current_cost;
}
double SimulatedAnnealing::calcSwapProb(int i, int j, double T, double swapCost)
{
   return std::min(1., std::exp(-(current.length + swapCost - best.length)/T));
}
bool SimulatedAnnealing::swap(int i, int j, double T)
{
   //std::cout << "TRY: calc swap cost"<< std::endl;// for debug
   double swapCost = calcSwapCost(i, j);
   //std::cout << "FINISH: calc swap cost"<< std::endl;// for debug
   //std::cout << "TRY: calc swap prob"<< std::endl;// for debug
   double swapProb = calcSwapProb(i, j, T, swapCost);
   //std::cout << "FINISH: calc swap prob"<< std::endl;// for debug
   if(rand(engine) < swapProb)
   {
      //std::cout << "i, j = " << current.tour[i] << ", " << current.tour[j] << std::endl;// for debug
      std::swap(current.tour[i], current.tour[j]);
      //std::cout << "i, j = " << current.tour[i] << ", " << current.tour[j] << std::endl;// for debug
      current.length += swapCost;
      if(is_bug())
      {
         std::cout << "i, j = " << i << ", " << j << std::endl;
         dispTour(current);
      }
      assert(current.tour[i] != current.tour[j]);
      return true;
   }
   return false;
}
double SimulatedAnnealing::solve(double T, double T_STOP, double C)
{
   // ここで初期化しても良い。
   // current = initialize_random();
   // best.tour.resize(ncity);
   // std::copy(current.tour.begin(), current.tour.end(), best.tour.begin());
   // best.length = current.length;
   int i, j;
   int iter = 0;
   bool success;
   while(T > T_STOP)
   {
      i = next_target(engine);
      j = next_target(engine);
      if(i == j) continue;
      //std::cout << "iter: " << iter << std::endl;// for debug
      ++iter;
      //std::cout << "try: swap"<< std::endl;// for debug
      success = swap(i, j, T);
      //std::cout << "finish: swap"<< std::endl;// for debug
      if(is_bug())
      {
         std::cout << "\niter: " << iter << ", T=" << T << "\ni, j = " << i << ", " << j << std::endl;
         dispTour(current);
         return -1;
      }
      if(success && (current.length < best.length))
      {
         best.tour.resize(ncity);
         std::copy(current.tour.begin(), current.tour.end(), best.tour.begin());
         best.length = current.length;
      }
      T *= C;
   }
   return best.length;
}