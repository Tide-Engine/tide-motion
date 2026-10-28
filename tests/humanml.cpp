#include "humanml.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace motion;
namespace h=motion::humanml;
void expect(bool v,const char* message) { if(!v)throw std::runtime_error(message); }
bool near(Vec3 a,Vec3 b,float tolerance=1e-5f) { Vec3 d=a-b;return dot(d,d)<tolerance*tolerance; }
template<class F> void refuses(F f) { bool caught=false;try {f();}catch(const std::runtime_error&){caught=true;}expect(caught,"expected refusal"); }
h::Row row() { h::Row r{};r[3]=1;for(size_t j=0;j<21;++j){r[67+6*j]=1;r[71+6*j]=1;}return r; }
int main(int argc,char** argv) {
    try {
        h::Statistics stats;for(size_t i=0;i<h::features;++i){stats.mean[i]=static_cast<float>(i)/100;stats.deviation[i]=0.5f+static_cast<float>(i)/50;}
        std::vector<h::Row> rows(5,row());
        auto normalized_rows=h::normalize(rows,stats);auto roundtrip=h::normalize(normalized_rows,stats,true);
        for(size_t f=0;f<rows.size();++f)for(size_t i=0;i<h::features;++i)
            expect(std::abs(rows[f][i]-roundtrip[f][i])<1e-5,"normalization roundtrip");
        auto bad=stats;bad.deviation[262]=0;refuses([&]{h::normalize(rows,bad);});
        rows[2][262]=std::numeric_limits<float>::infinity();refuses([&]{h::roots(rows);});rows[2][262]=0;
        // Analytic 90-degree turn: previous velocity is integrated in the CURRENT
        // frame's inverse heading, and the angular feature is a HALF angle.
        rows[0][0]=3.14159265358979323846f/4;rows[0][1]=1;rows[1][2]=2;
        auto roots=h::roots(rows);
        expect(near(roots[0].position,{0,1,0}),"root origin/height");
        expect(near(roots[1].position,{0,1,1}),"root inverse heading and shifted velocity");
        expect(near(roots[2].position,{-2,1,1}),"root accumulation");
        rows[1][4]=1;rows[1][5]=0.25f;rows[1][6]=0;
        auto xyz=h::positions(rows);expect(near(xyz[1][1],{0,0.25f,2}),"RIC height is absolute, x/z relative");
        // Independent single-chain FK oracle, hand-authored metre offsets, no
        // dataset skeleton. Test rotation on the incoming edge (child 4).
        rows.assign(3,row());h::Positions offsets{};offsets[1]={1,0,0};offsets[4]={0,-1,0};offsets[7]={0,-1,0};
        const size_t i=67+3*6;rows[0][i]=0;rows[0][i+1]=1;rows[0][i+2]=0;
        rows[0][i+3]=-1;rows[0][i+4]=0;rows[0][i+5]=0;
        auto fk=h::rotation_fk(rows,offsets);
        expect(near(fk[0][1],{1,1,0})&&near(fk[0][4],{2,1,0})&&near(fk[0][7],{3,1,0}),"incoming-edge FK oracle");
        // Shoulder chains reset orientation to root rather than spine3.
        offsets[3]={0,1,0};offsets[6]={0,1,0};offsets[9]={0,1,0};offsets[13]={1,0,0};
        rows[0][67+2*6]=0;rows[0][68+2*6]=1;rows[0][70+2*6]=-1;rows[0][71+2*6]=0;
        fk=h::rotation_fk(rows,offsets);
        expect(near(fk[0][13]-fk[0][9],{1,0,0}),"upstream collar chain convention");
        rows[0][67]=0;refuses([&]{h::rotation_fk(rows,offsets);});
        // Canonical rest and an articulated leg authored using Tide FK. RIC
        // samples are built from independently evaluated joint positions.
        Clip source;source.joints=soma();source.fps=20;
        for(size_t f=0;f<5;++f)for(auto& j:source.joints)source.poses.push_back(j.rest);
        for(size_t f=0;f<5;++f)source.frame(f)[0].p={static_cast<float>(f)*0.1f,1,0};
        source.frame(2)[23].q={0,0,std::sin(0.25f),std::cos(0.25f)};
        constexpr size_t mapped[22]={0,22,26,1,23,27,2,24,28,3,25,29,5,10,16,6,11,17,12,18,13,19};
        rows.assign(5,row());
        for(size_t f=0;f<5;++f) {
            auto w=world_pose(source,source.frame(f));rows[f][1]=0.1f;
            for(size_t j=1;j<22;++j) {auto p=w[mapped[j]].p;p.x-=w[0].p.x;p.z-=w[0].p.z;
                rows[f][4+(j-1)*3]=p.x;rows[f][5+(j-1)*3]=p.y;rows[f][6+(j-1)*3]=p.z;}
        }
        auto canonical=h::to_soma(rows);require_soma(canonical);
        expect(canonical.fps==30&&canonical.frames()==7,"20 to 30 Hz duration/sample count");
        expect(near(canonical.frame(6)[0].p,{0.4f,1,0}),"root resampling end");
        auto w=world_pose(canonical,canonical.frame(3));auto expected=world_pose(source,source.frame(2));
        expect(near(normalized(w[24].p-w[23].p),normalized(expected[24].p-expected[23].p)),"articulated knee direction");
        for(size_t f=0;f<canonical.frames();++f)for(size_t j=1;j<30;++j)
            expect(near(canonical.frame(f)[j].p,canonical.joints[j].rest.p),"canonical lengths fixed");
        auto edited=canonical;edited.frame(3)[0].p.x+=0.02f;
        auto spliced=h::splice(canonical,edited,2,5);
        for(size_t f=0;f<canonical.frames();++f)for(size_t j=0;j<30;++j) {
            auto a=spliced.frame(f)[j],b=(f>=2&&f<5?edited:canonical).frame(f)[j];
            expect(a.p.x==b.p.x&&a.p.y==b.p.y&&a.p.z==b.p.z&&a.q.x==b.q.x&&a.q.y==b.q.y&&a.q.z==b.q.z&&a.q.w==b.q.w,"splice exact preservation");
        }
        edited.frame(2)[0].p.x+=0.01f;refuses([&]{h::splice(canonical,edited,2,5);});
        refuses([&]{h::splice(canonical,canonical,5,2);});
        expect(h::mask_count(49,0,18)==49&&h::mask_count(49,17,18)==1,"schedule endpoints");
        for(size_t n=1;n<=49;++n)for(size_t t=1;t<18;++t)
            expect(h::mask_count(n,t,18)<=h::mask_count(n,t-1,18),"schedule monotone");
        refuses([&]{h::mask_count(0,0,18);});
        if(argc==2) { save_glb(canonical,utf8path(argv[1]));auto loaded=load_glb(utf8path(argv[1]));require_soma(loaded);expect(loaded.frames()==7,"GLB roundtrip"); }
        std::cout<<"HumanML native source-only contracts passed (no model inference)\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
