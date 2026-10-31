#pragma once

void toneTests(){
    auto pos=positions();ToneLocator locator;
    auto signal=[&](Vec target,double frequency,double amplitude,bool crowd){
        std::vector<float> audio(BLOCK*CH,0);
        auto add=[&](Vec source,double f,double level,double phase){
            double norm=sqrt(source.x*source.x+source.y*source.y+1);
            for(int ch=0;ch<CH;ch++){
                double delay=(pos[ch].x*source.x+pos[ch].y*source.y)/(norm*SPEED);
                Complex oscillator=std::polar(1.,2*PI*f*delay+phase),step=std::polar(1.,2*PI*f/FS);
                for(int t=0;t<BLOCK;t++,oscillator*=step)audio[t*CH+ch]+=(float)(level*oscillator.imag());
            }
        };
        add(target,frequency,amplitude,.83);
        if(crowd){
            const Vec voices[]={{-1.1,.2,1},{.8,-.6,1},{.1,.8,1}};
            const double fundamentals[]={133,169,203};
            for(int v=0;v<3;v++)for(int harmonic=1;harmonic<=16;harmonic++)add(voices[v],fundamentals[v]*harmonic,.012/sqrt((double)harmonic),v+harmonic*harmonic*.38);
            std::mt19937 random(714);std::normal_distribution<double> noise(0,.003);
            for(auto& sample:audio)sample+=(float)noise(random);
        }
        return audio;
    };
    auto begin=Clock::now();int count=0;
    for(Vec target:std::vector<Vec>{{0,0,1},{.66,.35,1},{-.88,-.47,1},{0,.7,1}})for(double frequency:{438.,440.,442.}){
        auto audio=signal(target,frequency,.008,true);auto result=locator.process(audio);count++;
        std::cout<<"Tone "<<frequency<<" Hz, source "<<target.x<<","<<target.y<<" -> "<<result.x<<","<<result.y<<" clarity="<<result.snr<<" dB residual="<<result.residual<<std::endl;
        require(result.valid&&std::abs(result.x-target.x)<.12&&std::abs(result.y-target.y)<.12,"Tone direction failed with louder synthetic competing voices");
    }
    for(double frequency:{400.,450.,480.,880.})require(!locator.process(signal({.5,.2,1},frequency,.03,false)).valid,"Wrong tone frequency accepted");
    require(!locator.process(signal({0,0,1},440,0,true)).valid,"Synthetic crowd alone produced a tone lock");
    require(!locator.process(std::vector<float>(BLOCK*CH,0)).valid,"Silence produced a tone lock");
    auto buried=signal({.5,.2,1},440,.0001,false);std::mt19937 random(216);std::normal_distribution<double> noise(0,.04);for(auto& sample:buried)sample+=(float)noise(random);
    require(!locator.process(buried).valid,"Tone below narrowband noise was accepted");
    auto inconsistent=signal({0,0,1},440,.04,false);for(int t=0;t<BLOCK;t++)for(int ch=0;ch<CH;ch++)inconsistent[t*CH+ch]=(float)(.04*sin(2*PI*440*t/FS+ch*1.73));
    require(!locator.process(inconsistent).valid,"Inconsistent microphone phases accepted");
    auto clipped=signal({0,0,1},440,.04,false);clipped[50]=1;require(!locator.process(clipped).valid,"Clipped audio accepted");
    auto invalid=clipped;invalid[50]=std::numeric_limits<float>::quiet_NaN();require(!locator.process(invalid).valid,"Nonfinite audio accepted");
    // End-to-end integration keeps the ordinary sound locator available.
    Beamformer beam;auto mixed=beam.process(signal({.66,.35,1},440,.008,true));
    require(mixed.tone.valid&&std::abs(mixed.tone.x-.66)<.12&&mixed.map.size()==GW*GH,"Beamformer did not expose separate tone and broadband results");
    auto benchmark=signal({.66,.35,1},440,.008,true);auto processingStart=Clock::now();for(int i=0;i<8;i++)beam.process(benchmark);
    std::cout<<"Mean combined broadband + tone processing: "<<std::chrono::duration<double,std::milli>(Clock::now()-processingStart).count()/8<<" ms per 171 ms block"<<std::endl;
    std::cout<<"PASS: "<<count<<" noisy tone directions, frequency tolerance, wrong-frequency/crowd/silence/phase/clipping rejection and beamformer integration. Test time: "<<std::chrono::duration<double>(Clock::now()-begin).count()<<" sec"<<std::endl;
}

void writePhoneTone(){
    // A transferable playback file, not a microphone recording. No audio device opened.
    const unsigned rate=48000,frames=rate*60,bytes=frames*2;
    std::ofstream out("440Hz-calibration.wav",std::ios::binary);
    auto u16=[&](unsigned n){out.put((char)(n&255));out.put((char)((n>>8)&255));};
    auto u32=[&](unsigned n){u16(n&65535);u16(n>>16);};
    out.write("RIFF",4);u32(36+bytes);out.write("WAVEfmt ",8);u32(16);u16(1);u16(1);u32(rate);u32(rate*2);u16(2);u16(16);out.write("data",4);u32(bytes);
    for(unsigned t=0;t<frames;t++){double fade=std::min({1.,t/(rate*.05),(frames-1-t)/(rate*.05)});short sample=(short)(.25*32767*fade*sin(2*PI*440*t/rate));u16((unsigned short)sample);}
    out.flush();require(bool(out),"Could not create phone tone WAV");std::cout<<"Created 440Hz-calibration.wav (60-second sine tone)."<<std::endl;
}
