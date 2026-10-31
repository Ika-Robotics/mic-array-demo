#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace setup {
struct Point { double x=0,y=0; };
struct Region {
    double left=.25,top=.15,right=.75,bottom=.85;
    bool valid() const {return std::isfinite(left)&&std::isfinite(top)&&std::isfinite(right)&&std::isfinite(bottom)&&left>=0&&top>=0&&right<=1&&bottom<=1&&right-left>=.12&&bottom-top>=.12;}
    bool contains(Point p) const {return p.x>=left&&p.x<=right&&p.y>=top&&p.y<=bottom;}
};
inline double quantile(std::vector<double> values,double q) {
    if(values.empty())return 0;
    std::sort(values.begin(),values.end());double i=std::clamp(q,0.,1.)*(values.size()-1);size_t n=(size_t)i;
    return values[n]+(values[std::min(n+1,values.size()-1)]-values[n])*(i-n);
}
struct Observation {Point direction,pixel;double spread=0;};
struct Projection {
    // Normalized image coordinates = affine map of the two acoustic ray slopes.
    std::array<double,6> a{.6,0,.5,0,-.6,.5};
    double residual=0;
    Point project(Point p) const {return {a[0]*p.x+a[1]*p.y+a[2],a[3]*p.x+a[4]*p.y+a[5]};}
    Point unproject(Point p) const {double det=a[0]*a[4]-a[1]*a[3];return {(a[4]*(p.x-a[2])-a[1]*(p.y-a[5]))/det,(-a[3]*(p.x-a[2])+a[0]*(p.y-a[5]))/det};}
    bool valid() const {
        for(double n:a)if(!std::isfinite(n)||std::abs(n)>10)return false;
        double det=a[0]*a[4]-a[1]*a[3];
        return std::isfinite(residual)&&residual>=0&&residual<=.08&&std::abs(det)>.005&&std::abs(det)<20;
    }
};
inline bool solve(double m[3][4],double result[3]) {
    for(int i=0;i<3;i++){
        int pivot=i;for(int j=i+1;j<3;j++)if(std::abs(m[j][i])>std::abs(m[pivot][i]))pivot=j;
        if(std::abs(m[pivot][i])<1e-7)return false;
        for(int k=i;k<4;k++)std::swap(m[i][k],m[pivot][k]);
        double scale=m[i][i];for(int k=i;k<4;k++)m[i][k]/=scale;
        for(int j=0;j<3;j++)if(j!=i){double f=m[j][i];for(int k=i;k<4;k++)m[j][k]-=f*m[i][k];}
    }
    for(int i=0;i<3;i++)result[i]=m[i][3];return true;
}
inline bool fit(const std::vector<Observation>& points,Projection& out,std::string& error) {
    if(points.size()<4){error="Capture four alignment points first.";return false;}
    double minX=1,maxX=0,minY=1,maxY=0;
    for(auto p:points){
        if(!std::isfinite(p.direction.x)||!std::isfinite(p.direction.y)||!std::isfinite(p.pixel.x)||!std::isfinite(p.pixel.y)||p.pixel.x<0||p.pixel.x>1||p.pixel.y<0||p.pixel.y>1){error="Invalid alignment coordinates.";return false;}
        minX=std::min(minX,p.pixel.x);maxX=std::max(maxX,p.pixel.x);minY=std::min(minY,p.pixel.y);maxY=std::max(maxY,p.pixel.y);
    }
    if(maxX-minX<.3||maxY-minY<.3){error="Spread the points farther apart across the camera view.";return false;}
    for(int axis=0;axis<2;axis++){
        double m[3][4]{};
        for(auto p:points){double v[3]={p.direction.x,p.direction.y,1},target=axis?p.pixel.y:p.pixel.x;
            for(int i=0;i<3;i++){for(int j=0;j<3;j++)m[i][j]+=v[i]*v[j];m[i][3]+=v[i]*target;}}
        double values[3];if(!solve(m,values)){error="Sound directions were too similar. Retry at separated positions.";return false;}
        for(int i=0;i<3;i++)out.a[axis*3+i]=values[i];
    }
    double sum=0;for(auto p:points){auto q=out.project(p.direction);sum+=(q.x-p.pixel.x)*(q.x-p.pixel.x)+(q.y-p.pixel.y)*(q.y-p.pixel.y);}
    out.residual=sqrt(sum/points.size());
    if(!out.valid()){error="Alignment is inconsistent. Retry with one steady sound and fewer reflections.";return false;}return true;
}
struct NoiseResult {bool good=false;double floor=-65,gate=-55,spread=0;std::array<double,16> channels{};std::string message;};
inline NoiseResult measureNoise(const std::vector<std::array<double,16>>& samples,const std::vector<double>& levels,bool clipped) {
    NoiseResult out;
    if(samples.size()<24||levels.size()!=samples.size()){out.message="Not enough fresh audio. Check the array and retry.";return out;}
    for(double value:levels)if(!std::isfinite(value)||value>0||value< -150){out.message="Invalid audio measurements. Retry hardware check.";return out;}
    for(auto row:samples)for(double value:row)if(!std::isfinite(value)||value>0||value< -150){out.message="Invalid microphone measurements.";return out;}
    if(clipped){out.message="Audio clipped during room measurement. Reduce nearby sound and retry.";return out;}
    out.floor=quantile(levels,.5);out.spread=quantile(levels,.9)-quantile(levels,.1);
    if(out.spread>8){out.message="The room changed too much. Keep quiet for the full measurement and retry.";return out;}
    for(int c=0;c<16;c++){std::vector<double> v;for(auto row:samples)v.push_back(row[c]);out.channels[c]=quantile(v,.5);if(out.channels[c]< -105){out.message="One or more microphones are silent. Return to the hardware check.";return out;}}
    out.gate=std::clamp(quantile(levels,.8)+6.,-90.,-15.);out.good=true;out.message="Room measured. The gate is 6 dB above the upper background level.";return out;
}
struct CaptureResult {bool good=false;Point median;double spread=0;std::string message;};
inline CaptureResult summarize(const std::vector<Point>& points,int total) {
    CaptureResult out;
    if(points.size()<10||total<10||(double)points.size()/total<.40){out.message="Too little clear sound. Try steady speech or rustling paper above the room noise.";return out;}
    std::vector<double> x,y,d;
    for(auto p:points){if(!std::isfinite(p.x)||!std::isfinite(p.y)){out.message="Invalid sound direction.";return out;}x.push_back(p.x);y.push_back(p.y);}
    out.median={quantile(x,.5),quantile(y,.5)};
    for(auto p:points)d.push_back(hypot(p.x-out.median.x,p.y-out.median.y));out.spread=quantile(d,.8);
    if(out.spread>.30){out.message="The sound moved or reflections dominated. Hold one source steady and retry.";return out;}
    out.good=true;out.message="Steady sound position captured.";return out;
}
struct Profile {
    Projection projection;Region region;double noise=-65,gate=-55,validation=1,insideRate=0,outsideRate=1;
    std::array<double,16> channels{};std::string audioId,cameraId;int width=0,height=0;
    bool valid() const {
        if(!projection.valid()||!region.valid()||!std::isfinite(noise)||!std::isfinite(gate)||noise< -105||noise> -5||gate< -90||gate> -5||!std::isfinite(validation)||validation<0||validation>.10)return false;
        if(!std::isfinite(insideRate)||!std::isfinite(outsideRate)||insideRate<.6||insideRate>1||outsideRate<0||outsideRate>.35)return false;
        for(double v:channels)if(!std::isfinite(v)||v< -105||v>0)return false;
        return !audioId.empty()&&!cameraId.empty()&&width>=160&&height>=120&&width<=8192&&height<=8192;
    }
    bool matches(const std::string& a,const std::string& c,int w,int h)const{return audioId==a&&cameraId==c&&width==w&&height==h;}
};
inline void serialize(std::ostream& out,const Profile& p) {
    out<<std::setprecision(17)<<"IKA_MIC_SETUP 1\n"<<std::quoted(p.audioId)<<'\n'<<std::quoted(p.cameraId)<<'\n'<<p.width<<' '<<p.height<<'\n';
    out<<p.noise<<' '<<p.gate<<' '<<p.validation<<' '<<p.insideRate<<' '<<p.outsideRate<<'\n';
    for(double n:p.channels)out<<n<<' ';out<<'\n';for(double n:p.projection.a)out<<n<<' ';out<<p.projection.residual<<'\n';
    out<<p.region.left<<' '<<p.region.top<<' '<<p.region.right<<' '<<p.region.bottom<<'\n';
}
inline bool deserialize(std::istream& in,Profile& result) {
    Profile p;std::string header;int version=0;
    if(!(in>>header>>version)||header!="IKA_MIC_SETUP"||version!=1)return false;
    if(!(in>>std::quoted(p.audioId)>>std::quoted(p.cameraId)>>p.width>>p.height>>p.noise>>p.gate>>p.validation>>p.insideRate>>p.outsideRate))return false;
    for(auto& n:p.channels)if(!(in>>n))return false;for(auto& n:p.projection.a)if(!(in>>n))return false;
    if(!(in>>p.projection.residual>>p.region.left>>p.region.top>>p.region.right>>p.region.bottom))return false;
    std::string extra;if(in>>extra)return false;if(!p.valid())return false;result=p;return true;
}
class Tracker {
public:
    Point point;bool active=false;double last=-1,lastGood=-1,goodSince=-1;
    void update(bool usable,Point measured,double now) {
        if(!usable){goodSince=-1;if(lastGood<0||now-lastGood>.3)active=false;return;}
        if(last<0||now-lastGood>.5){point=measured;goodSince=now;active=false;}
        else {double dt=std::clamp(now-last,.01,.5),alpha=1-exp(-dt/.18);point.x+=alpha*(measured.x-point.x);point.y+=alpha*(measured.y-point.y);}
        if(goodSince<0)goodSince=now;if(now-goodSince>=.15)active=true;last=lastGood=now;
    }
    void reset(){*this=Tracker{};}
};
}
