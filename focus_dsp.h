#pragma once

namespace focus {
struct Geometry {
    double hfov=80,aspect=4./3.;int rotation=0;bool flipX=false,flipY=false;
    Vec orient(Vec p)const{for(int i=0;i<rotation;i++){double x=p.x;p.x=-p.y;p.y=x;}if(flipX)p.x=-p.x;if(flipY)p.y=-p.y;return p;}
    Vec unorient(Vec p)const{if(flipX)p.x=-p.x;if(flipY)p.y=-p.y;for(int i=0;i<rotation;i++){double x=p.x;p.x=p.y;p.y=-x;}return p;}
    Vec ray(double x,double y)const{double h=tan(hfov*PI/360);return unorient({(2*x-1)*h,(1-2*y)*h/aspect,1});}
    setup::Point project(Vec p)const{p=orient(p);double h=tan(hfov*PI/360);return {.5*(p.x/h+1),.5*(1-p.y*aspect/h)};}
};

// Second-order Butterworth sections. Identical filters are available on the
// reference channel so spatial improvement can be measured separately from EQ.
struct Biquad {
    double b0=0,b1=0,b2=0,a1=0,a2=0,z1=0,z2=0;
    Biquad(double frequency,bool highpass){
        double w=2*PI*frequency/FS,cs=cos(w),alpha=sin(w)/sqrt(2.),a0=1+alpha;
        b0=(highpass?1+cs:1-cs)/(2*a0);b1=(highpass?-(1+cs):1-cs)/a0;b2=b0;a1=-2*cs/a0;a2=(1-alpha)/a0;
    }
    double process(double x){double y=b0*x+z1;z1=b1*x-a1*y+z2;z2=b2*x-a2*y;return y;}
};
struct SpeechBand {
    Biquad high{150,true},low{3800,false};
    double process(double sample){return low.process(high.process(sample));}
};
struct Output {std::vector<float> original,reference,focused;double originalDb=-120,focusedDb=-120;};
inline double level(const std::vector<float>& samples){double sum=0;for(float v:samples)sum+=(double)v*v;return 10*log10(std::max(sum/std::max(size_t(1),samples.size()),1e-16));}
class Beam {
    static constexpr int History=128,Taps=9,CenterDelay=32;
    std::array<std::array<double,History>,CH> history{};
    std::array<std::array<double,Taps>,CH> weights{};
    std::array<int,CH> starts{};int cursor=0;Vec previous{999,999,1};
    SpeechBand focusedBand,referenceBand;
public:
    void reset(){*this=Beam{};}
    Output process(const std::vector<float>& audio,Vec ray){
        if(audio.size()%CH)throw std::runtime_error("Incomplete microphone frame");
        if(!std::isfinite(ray.x)||!std::isfinite(ray.y))throw std::runtime_error("Invalid focus direction");
        if(ray.x!=previous.x||ray.y!=previous.y){
            double norm=sqrt(ray.x*ray.x+ray.y*ray.y+1);auto pos=positions();
            for(int c=0;c<CH;c++){
                double delay=CenterDelay+FS*(pos[c].x*ray.x+pos[c].y*ray.y)/(SPEED*norm);
                starts[c]=(int)floor(delay)-Taps/2;double sum=0;
                for(int j=0;j<Taps;j++){double x=starts[c]+j-delay;double sinc=std::abs(x)<1e-10?1:sin(PI*x)/(PI*x);weights[c][j]=sinc*(.5+.5*cos(PI*x/(Taps/2+1.)));sum+=weights[c][j];}
                for(auto& w:weights[c])w/=sum;
            }
            previous=ray;
        }
        Output out;size_t frames=audio.size()/CH;out.original.resize(frames);out.reference.resize(frames);out.focused.resize(frames);
        for(size_t t=0;t<frames;t++){
            for(int c=0;c<CH;c++){float v=audio[t*CH+c];if(!std::isfinite(v))throw std::runtime_error("Nonfinite microphone input");history[c][cursor]=v;}
            double sum=0;
            for(int c=0;c<CH;c++)for(int j=0;j<Taps;j++)sum+=weights[c][j]*history[c][(cursor-starts[c]-j+History)%History]/CH;
            double original=history[0][(cursor-CenterDelay+History)%History];
            out.original[t]=(float)original;out.reference[t]=(float)referenceBand.process(original);out.focused[t]=(float)focusedBand.process(sum);
            cursor=(cursor+1)%History;
        }
        out.originalDb=level(out.original);out.focusedDb=level(out.focused);return out;
    }
};
}
