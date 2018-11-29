#ifndef M_COMPLEX
#define M_COMPLEX

class Complex{
public:
  double x;
  double y;
  __host__ __device__ Complex(double _x = 0, double _y = 0):x(_x), y(_y){}
  __host__ __device__ Complex(const Complex &other){
      x = other.x;
      y = other.y;
  }
};

__device__ double complexAbs(Complex const &c){
  double result = sqrt(c.x*c.x + c.y*c.y);
  return result;
}

__device__ Complex addComplex(Complex const &c1, Complex const &c2){
  double real = c1.x + c2.x;
  double imaginary = c1.y + c2.y;
  Complex result(real, imaginary);
  return result;
}

__device__ Complex multiplyComplex(Complex const &c1, Complex const &c2){
  double real = c1.x * c2.x - c1.y * c2.y;
  double imaginary = c1.x * c2.y + c1.y * c2.x;
  Complex result(real, imaginary);
  return result;
}

#endif
