// Source-only numerical feasibility probe. No trained model, body data or inference.
// DART source references and limits: notes/dart-feasibility.md.
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using V = std::array<double, 3>;
using M = std::array<V, 3>; // rows, acting on column vectors
V add(V a, V b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
V mul(V a, double s) { return {a[0]*s, a[1]*s, a[2]*s}; }
double dot(V a, V b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
V cross(V a, V b) { return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
V unit(V a) {
    const double n = std::sqrt(dot(a,a));
    if (!std::isfinite(n) || n < 1e-10) throw std::runtime_error("degenerate 6D rotation");
    return mul(a,1/n);
}
V apply(M r, V v) { return {dot(r[0],v),dot(r[1],v),dot(r[2],v)}; }
M transpose(M r) {
    return {{{r[0][0],r[1][0],r[2][0]}, {r[0][1],r[1][1],r[2][1]}, {r[0][2],r[1][2],r[2][2]}}};
}
M rows6(V a, V b) {
    a=unit(a); b=unit(add(b,mul(a,-dot(a,b))));
    return {a,b,cross(a,b)};
}
void near(double a, double b, double eps=1e-10) {
    if (!std::isfinite(a) || std::abs(a-b)>eps) throw std::runtime_error("numerical mismatch");
}
void near(V a, V b) { for (int i=0;i<3;++i) near(a[i],b[i]); }

// eta=0 DDIM with an x0 predictor: x_prev=A*x+B*x0. This is only
// the sampler's local adjoint; DART also needs denoiser/decoder/history VJPs.
struct Step { double value, dx, dx0; };
Step ddim(double x, double x0, double alpha, double prev) {
    const double a=std::sqrt((1-prev)/(1-alpha));
    const double b=std::sqrt(prev)-a*std::sqrt(alpha);
    return {a*x+b*x0,a,b};
}

int main() try {
    // Independent +90-degree Z rotation fixture catches row/column confusion.
    const M r=rows6({0,-2,0},{3,4,0});
    near(apply(r,{1,0,0}),V{0,1,0});
    near(dot(r[0],r[1]),0); near(dot(r[2],cross(r[0],r[1])),1);
    for (const auto bad : {V{0,0,0},V{0,-1,0}}) {
        bool refused=false;
        try { (void)rows6({0,-2,0},bad); } catch (const std::runtime_error&) { refused=true; }
        if (!refused) throw std::runtime_error("accepted degenerate rows");
    }
    // Pelvis offset must be rotated before subtraction; plain R*t+T is wrong.
    const V t{1,2,3}, delta{.2,-.1,.4}, origin{5,-2,1};
    const V world=add(add(apply(r,add(t,delta)),mul(delta,-1)),origin);
    near(world,V{2.9,-.7,4});
    const V local=add(apply(transpose(r),add(add(world,delta),mul(origin,-1))),mul(delta,-1));
    near(local,t);
    // Proper Z-up -> Y-up basis (x,y,z)->(x,z,-y); preserves handedness.
    const M basis{{{1,0,0},{0,0,1},{0,-1,0}}};
    near(apply(basis,{0,0,1}),V{0,1,0});
    near(cross(apply(basis,{1,0,0}),apply(basis,{0,1,0})),apply(basis,{0,0,1}));
    double worst=0;
    for (double alpha : {.05,.4,.9}) for (double prev : {.95,1.0}) {
        const double x=.3,x0=-.2,h=1e-6;
        const auto s=ddim(x,x0,alpha,prev);
        const double dx=(ddim(x+h,x0,alpha,prev).value-ddim(x-h,x0,alpha,prev).value)/(2*h);
        const double dx0=(ddim(x,x0+h,alpha,prev).value-ddim(x,x0-h,alpha,prev).value)/(2*h);
        near(s.dx,dx,1e-8); near(s.dx0,dx0,1e-8);
        worst=std::fmax(worst,std::fmax(std::abs(s.dx-dx),std::abs(s.dx0-dx0)));
        if (prev==1) near(s.value,x0);
    }
    std::cout << "PASS: row-6D, degeneracy, pelvis/world inverse, handedness; DDIM local adjoint max error " << worst << '\n';
    std::cout << "No DART inference, ONNX export, SOMA mapping or goal accuracy validated.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
}
