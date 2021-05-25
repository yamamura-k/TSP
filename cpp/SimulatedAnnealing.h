#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <limits>

#ifndef rad2deg(a)
#define rad2deg(a) ((a)/M_PI * 180.0) /* rad を deg に換算するマクロ関数 */
#endif

#ifndef deg2rad(a)
#define deg2rad(a) ((a)/180.0 * M_PI) /* deg を rad に換算するマクロ関数 */
#endif

struct Coordinate
{
    double x;
    double y;
};
typedef std::vector< std::vector<double> > MatrixDouble;
typedef std::vector<double> VectorDouble;
typedef std::map<int, Coordinate> Coord;
typedef double (*FUNCTYPE)(Coordinate A, Coordinate B);

const double earth_r = 6378.137;
//constexpr double EPS = std::numeric_limits<double>::epsilon();
const double EPS = 1e-8;
double calcDistGeo(Coordinate A, Coordinate B);
double calcDistEUC2D(Coordinate A, Coordinate B);
int read(std::string filename, int& ncity, MatrixDouble& D, Coord& coord);
struct Tour
{
   double length;
   std::vector<int> tour;
};

class SimulatedAnnealing
{
   private:
      double calcSwapProb(int i, int j, double T, double swapCost);
      bool swap(int i, int j, double T);
   protected:
      double calcSwapCost(int i, int j);
      bool is_bug();
   public:
      Tour best;
      Tour current;
      int ncity;
      MatrixDouble D;
      std::random_device seed_gen;
      std::mt19937 engine;
      std::uniform_real_distribution<> rand;
      std::uniform_int_distribution<> next_target;
      SimulatedAnnealing(int _ncity, MatrixDouble _D);
      Tour initialize_random();
      void dispTour(Tour tmp);
      double calcTourLen(Tour tmp);
      double solve(double T, double T_STOP, double C);
};