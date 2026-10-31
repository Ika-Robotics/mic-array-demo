#pragma once

struct ToneEstimate {
    bool valid=false;
    double x=0,y=0,frequency=440,level=-120,snr=0,residual=10;
};

// A separate narrowband phase fit: broadband PHAT removes magnitude and would
// otherwise amplify unrelated crowd noise while excluding the 440 Hz source.
// At 440 Hz the array's full diagonal is less than half a wavelength, so pair
// phases need no unwrapping. As with the broadband estimator, assume front side.
class ToneLocator {
    std::array<Vec,CH> pos=positions();
    std::vector<double> window;
    double windowSum=0;
public:
    ToneLocator():window(BLOCK){for(int t=0;t<BLOCK;t++){window[t]=.5-.5*cos(2*PI*t/(BLOCK-1));windowSum+=window[t];}}
    ToneEstimate process(const std::vector<float>& audio){
        ToneEstimate out;
        if(audio.size()!=BLOCK*CH)return out;
        std::vector<std::vector<Complex>> spectra(CH,std::vector<Complex>(BLOCK));
        std::vector<double> power(BLOCK/2,0);
        for(int c=0;c<CH;c++){
            for(int t=0;t<BLOCK;t++){double v=audio[t*CH+c];if(!std::isfinite(v)||std::abs(v)>.99)return out;spectra[c][t]=v*window[t];}
            fft(spectra[c]);for(int k=1;k<BLOCK/2;k++)power[k]+=std::norm(spectra[c][k])/CH;
        }
        int first=(int)std::floor(428*BLOCK/FS),last=(int)std::ceil(452*BLOCK/FS),peak=first;
        for(int k=first+1;k<=last;k++)if(power[k]>power[peak])peak=k;
        // Log-power interpolation reduces FFT-bin bias for an unsynchronized phone.
        double a=log(std::max(power[peak-1],1e-30)),b=log(std::max(power[peak],1e-30)),c=log(std::max(power[peak+1],1e-30));
        double denominator=a-2*b+c,offset=std::abs(denominator)>1e-12?std::clamp(.5*(a-c)/denominator,-.5,.5):0;
        out.frequency=(peak+offset)*FS/BLOCK;
        std::array<Complex,CH> phasor{};
        Complex oscillator=1,step=std::polar(1.,-2*PI*out.frequency/FS);
        for(int t=0;t<BLOCK;t++,oscillator*=step)for(int ch=0;ch<CH;ch++)phasor[ch]+=(double)audio[t*CH+ch]*window[t]*oscillator;
        std::array<bool,CH> good{};int goodChannels=0;double signal=0,background=0;
        for(int ch=0;ch<CH;ch++){
            std::vector<double> nearby;
            for(int delta=6;delta<=12;delta++){nearby.push_back(std::norm(spectra[ch][peak-delta]));nearby.push_back(std::norm(spectra[ch][peak+delta]));}
            std::sort(nearby.begin(),nearby.end());double noise=std::max(nearby[nearby.size()/2],1e-18),tone=std::norm(phasor[ch]);
            good[ch]=tone/noise>=10.;goodChannels+=good[ch];signal+=tone; background+=noise;
        }
        out.level=10*log10(std::max(2*signal/(CH*windowSum*windowSum),1e-16));
        out.snr=10*log10(std::max(signal,1e-18)/std::max(background,1e-18));
        if(std::abs(out.frequency-440)>4||out.level< -90||out.snr<12||goodChannels<12)return out;
        double xx=0,xy=0,yy=0,xp=0,yp=0,wave=2*PI*out.frequency/SPEED;
        for(int i=0;i<CH;i++)for(int j=i+1;j<CH;j++)if(good[i]&&good[j]){
            double dx=wave*(pos[i].x-pos[j].x),dy=wave*(pos[i].y-pos[j].y),phase=std::arg(phasor[i]*std::conj(phasor[j]));
            xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;xp+=dx*phase;yp+=dy*phase;
        }
        double det=xx*yy-xy*xy;if(det<1e-9)return out;
        double ux=(xp*yy-yp*xy)/det,uy=(yp*xx-xp*xy)/det,error=0;int pairs=0;
        for(int i=0;i<CH;i++)for(int j=i+1;j<CH;j++)if(good[i]&&good[j]){
            double expected=wave*((pos[i].x-pos[j].x)*ux+(pos[i].y-pos[j].y)*uy);
            double residual=std::remainder(std::arg(phasor[i]*std::conj(phasor[j]))-expected,2*PI);error+=residual*residual;pairs++;
        }
        out.residual=sqrt(error/std::max(1,pairs));double radial=ux*ux+uy*uy;
        if(out.residual>.18||radial>=.94)return out;
        double uz=sqrt(1-radial);out.x=ux/uz;out.y=uy/uz;out.valid=true;return out;
    }
};
